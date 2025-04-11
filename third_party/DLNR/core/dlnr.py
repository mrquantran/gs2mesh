import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import LSTMMultiUpdateBlock
from core.extractor import ResidualBlock, Channel_Attention_Transformer_Extractor
from core.corr import CorrBlock1D, CorrBlockFast1D
from core.utils.utils import coords_grid, upflow
from nets.refinement import *
import math # Import math for log

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class DLNR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        context_dims = args.hidden_dims
        self.extractor = Channel_Attention_Transformer_Extractor()  #
        self.update_block = LSTMMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.bias_convs = nn.ModuleList(
            [nn.Conv2d(context_dims[i], args.hidden_dims[i] * 4, 3, padding=3 // 2) for i in
             range(self.args.n_gru_layers)])

        self.volume_conv = nn.Sequential(
            ResidualBlock(128, 128, 'instance', stride=1),
            nn.Conv2d(128, 256, 3, padding=1))

        self.normalizationRefinement = NormalizationRefinement()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask): # mask is the raw upsampling weights
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        # factor should be an integer, typically 4 if n_downsample=2

        # --- Safety Check: Sanitize input mask ---
        if mask is not None: # Check if mask is None first
            if torch.isnan(mask).any() or torch.isinf(mask).any():
                print(f"!!! Warning: NaN/Inf detected in mask input to upsample_flow. Sanitizing again.")
                mask = torch.nan_to_num(mask, nan=0.0, posinf=20.0, neginf=-20.0)
        else:
             # Handle case where mask might be None if update_block doesn't provide it
             print("!!! Warning: mask is None in upsample_flow input.")
             # Create a default uniform mask if needed, although confidence calculation will fail later
             # This part depends on how None mask should be handled upstream. Erroring might be better.
             # For now, let's assume mask is always provided when this function is called based on forward logic
             pass
        # --- End Safety Check ---


        # --- Stabilize Softmax Input ---
        clamp_min = -20.0
        clamp_max = 20.0
        mask_weights_raw = mask
        mask_weights_clamped = torch.clamp(mask_weights_raw, min=clamp_min, max=clamp_max)
        # --- End Stabilization ---

        # Reshape mask for softmax and confidence calculation
        # Assuming input mask has shape [N, 9 * factor * factor, H, W] or similar that allows this view
        try:
            # This view operation is crucial and depends on the exact output shape of update_block's up_mask
            mask_weights_clamped = mask_weights_clamped.view(N, 1, 9, factor, factor, H, W)
        except RuntimeError as e:
            print(f"!!! Error reshaping mask in upsample_flow. Original mask shape: {mask.shape}")
            print(f"Target view shape: {(N, 1, 9, factor, factor, H, W)}")
            raise e

        mask_weights = torch.softmax(mask_weights_clamped, dim=2) # Softmax over the 9 neighbors

        # --- Calculate Confidence ---
        entropy = -torch.sum(mask_weights * torch.log(mask_weights + 1e-9), dim=2) # Sum over neighbors dim -> [N, 1, factor, factor, H, W]

        if torch.isnan(entropy).any() or torch.isinf(entropy).any():
            entropy = torch.nan_to_num(entropy, nan=math.log(9.0), posinf=math.log(9.0), neginf=0.0)

        max_entropy = math.log(9.0)
        normalized_entropy = entropy / (max_entropy + 1e-9)
        confidence = torch.clamp(1.0 - normalized_entropy, min=0.0, max=1.0) # Shape: [N, 1, factor, factor, H, W]

        # <<< FIX TILING HERE using pixel_shuffle >>>
        confidence = confidence.squeeze(1) # Shape: [N, factor, factor, H, W]
        # Reshape confidence for pixel_shuffle: needs (N, C * factor^2, H, W) -> (N, factor*factor, H, W)
        try:
             confidence = confidence.view(N, factor*factor, H, W)
        except RuntimeError as e:
            print(f"!!! Error reshaping confidence for pixel_shuffle. Shape before view: {confidence.shape}")
            print(f"Target view shape: {(N, factor*factor, H, W)}")
            raise e

        # Apply pixel_shuffle
        if isinstance(factor, int) and factor > 0:
             confidence_shuffled = F.pixel_shuffle(confidence, factor) # Output: [N, 1, H*factor, W*factor]
        else:
             print(f"!!! Error: Upsampling factor ({factor}) is not a positive integer, cannot use pixel_shuffle.")
             confidence_shuffled = torch.ones(N, 1, H * factor, W * factor, device=flow.device) * 0.5 # Default fallback

        confidence_final = confidence_shuffled # Final shape: [N, 1, H_full, W_full]
        # <<< END FIX >>>

        # --- Upsample Flow ---
        # Prepare unfolded flow neighbors
        up_flow_unfolded = F.unfold(factor * flow, [3, 3], padding=1) # Shape [N, D*9, H*W]
        up_flow_neighbors = up_flow_unfolded.view(N, D, 9, 1, 1, H, W) # Shape [N, D, 9, 1, 1, H, W]

        # --- CORRECTED WEIGHT APPLICATION ---
        # Multiply weights (broadcasted) with flow neighbors and sum over neighbors
        # mask_weights shape: [N, 1, 9, factor, factor, H, W]
        # up_flow_neighbors shape: [N, D, 9, 1, 1, H, W]
        # Broadcasting happens automatically:
        # dim 1 of mask_weights expands to D
        # dim 3,4 of up_flow_neighbors expand to factor, factor
        product = mask_weights * up_flow_neighbors # Result shape: [N, D, 9, factor, factor, H, W]
        up_flow_summed = torch.sum(product, dim=2) # Sum over neighbors (dim 2, size 9)
                                                   # Result shape: [N, D, factor, factor, H, W]
        # --- END CORRECTED WEIGHT APPLICATION ---


        # --- Reshape summed flow to full resolution using permute + reshape ---
        # This permute/reshape combination IS standard for reconstructing from weighted sums
        # Need to carefully check the order based on how factor, factor, H, W correspond to output pixels
        # Typical order might be N, D, H, factor, W, factor -> N, D, H*factor, W*factor
        # Let's try the most common permute for this: (0, 1, 4, 2, 5, 3) -> N, D, H, factor, W, factor
        try:
            up_flow_permuted = up_flow_summed.permute(0, 1, 4, 2, 5, 3) # Shape: [N, D, H, factor, W, factor]
        except Exception as e:
             print(f"!!! Error permuting up_flow_summed. Shape was: {up_flow_summed.shape}")
             raise e

        # Now reshape to final size
        try:
             up_flow_final = up_flow_permuted.reshape(N, D, factor * H, factor * W) # Shape: [N, D, H_full, W_full]
        except RuntimeError as e:
            print(f"!!! Error reshaping up_flow_permuted. Shape was: {up_flow_permuted.shape}")
            print(f"Target reshape: {(N, D, factor * H, factor * W)}")
            raise e

        return up_flow_final, confidence_final # Return the correctly calculated flow and pixel_shuffled confidence


    # Modified forward to handle and return confidence
    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        with autocast(enabled=self.args.mixed_precision):
            *cnet_list, x = self.extractor(torch.cat((image1, image2), dim=0))
            fmap1, fmap2 = self.volume_conv(x).split(dim=0, split_size=x.shape[0] // 2)
            net_h = [torch.tanh(x[0]) for x in cnet_list]
            net_ext = [torch.relu(x[1]) for x in cnet_list]
            net_ext = [list(conv(i).split(split_size=conv.out_channels // 4, dim=1)) for i, conv in
                       zip(net_ext, self.bias_convs)]

        if self.args.corr_implementation == "reg":
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "reg_cuda":
            corr_block = CorrBlockFast1D
        corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        coords0, coords1 = self.initialize_flow(net_h[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        confidence_map = None # Initialize confidence map

        cnt = 0
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                if cnt == 0:
                    netC = net_h
                    cnt = 1
                # Get outputs from update block
                netC, net_h, up_mask, delta_flow = self.update_block(netC, net_h, net_ext, corr, flow,
                                                                     iter32=self.args.n_gru_layers == 3,
                                                                     iter16=self.args.n_gru_layers >= 2)

            # <<< Sanitize Network Outputs >>>
            if up_mask is not None:
                if torch.isnan(up_mask).any() or torch.isinf(up_mask).any():
                    print(f"!!! Warning: NaN/Inf detected in up_mask at iter {itr}. Sanitizing.")
                    up_mask = torch.nan_to_num(up_mask, nan=0.0, posinf=20.0, neginf=-20.0)

            if delta_flow is not None: # Check if delta_flow exists
                if torch.isnan(delta_flow).any() or torch.isinf(delta_flow).any():
                    print(f"!!! Warning: NaN/Inf detected in delta_flow at iter {itr}. Sanitizing.")
                    delta_flow = torch.nan_to_num(delta_flow, nan=0.0, posinf=1.0, neginf=-1.0)
                # Apply delta_flow
                delta_flow[:, 1] = 0.0 # Set vertical component to 0
                coords1 = coords1 + delta_flow
            else:
                 # Handle case where delta_flow might be None if update block doesn't produce it
                 print(f"!!! Warning: delta_flow is None at iter {itr}.")


            # Only compute full-res disparity and confidence towards the end or if testing
            if itr == iters - 1 or test_mode:
                # Calculate low-res flow *before* upsampling
                current_low_res_flow = coords1 - coords0

                if up_mask is None:
                    # Fallback if no upsampling mask is provided
                    print("!!! Warning: up_mask is None. Using simple upflow (confidence=1).")
                    disp_fullres = upflow(current_low_res_flow) # Use current low-res flow
                    N_full, _, H_full, W_full = disp_fullres.shape
                    confidence_map = torch.ones(N_full, 1, H_full, W_full, device=disp_fullres.device)
                else:
                    # Perform upsampling using the dedicated function
                    disp_fullres, confidence_map = self.upsample_flow(current_low_res_flow, up_mask) # Use current low-res flow

                disp_fullres = disp_fullres[:, :1] # Keep only disparity dimension (x-component)

                # Optional: Apply refinement
                if itr == iters - 1 and not test_mode:
                    # Ensure image1/image2 have correct shape if using mixed precision context outside
                    image1_f = (image1 + 1.0) / 2.0 * 255.0 # Example: Assuming image1 is [-1, 1] float
                    image2_f = (image2 + 1.0) / 2.0 * 255.0 # Example: Assuming image2 is [-1, 1] float
                    if disp_fullres.max() < 0:
                        with autocast(enabled=self.args.mixed_precision): # Ensure refinement is also under autocast if needed
                             refine_value = self.normalizationRefinement(disp_fullres, image1_f, image2_f) # Pass correctly scaled images
                             disp_fullres = disp_fullres + refine_value
                    # else: pass

            # Store prediction for training mode
            if not test_mode:
                # Ensure disp_fullres exists even if not last iteration (e.g., by moving calculation up)
                # For now, assumes list can contain None until last iter
                if itr == iters - 1: # Store only final prediction if intermediate ones aren't needed
                     flow_predictions.append(disp_fullres)
                # else: flow_predictions.append(None) # Or compute low-res if needed

        if test_mode:
            # Need final low-res flow (coords1 - coords0) and final full-res disp/confidence
            final_low_res_flow = coords1 - coords0
            # Ensure disp_fullres and confidence_map hold the final values from the loop
            if confidence_map is None: # If loop finished before final calc (e.g., iters=0)
                final_disp_fullres, final_confidence_map = self.upsample_flow(final_low_res_flow, up_mask) # Calculate final here
                final_disp_fullres = final_disp_fullres[:,:1]
            else:
                final_disp_fullres = disp_fullres
                final_confidence_map = confidence_map

            return final_low_res_flow, final_disp_fullres, final_confidence_map

        # Return list of full-res disparities during training
        return flow_predictions, confidence_map # Return final confidence map if needed