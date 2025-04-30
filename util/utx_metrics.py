import torch

from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure

def __compute_metrics(t_real : torch.Tensor, t_fake : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    t_real = torch.unsqueeze(t_real, dim = 0)
    t_fake = torch.unsqueeze(t_fake, dim = 0)

    ssim_x = structural_similarity_index_measure(t_fake, t_real).item()
    psnr_x = peak_signal_noise_ratio(t_fake, t_real).item()

    return ssim_x, psnr_x


def compute_single_batch_pix7mask(eval_instaces : dict[str, torch.Tensor], batch_index : int) -> dict[str, float]:
    real_b = eval_instaces['real_B']
    fake_b = eval_instaces['fake_B']
    info_i = eval_instaces['info_i']

    batch_mask_info = info_i[batch_index] # (2, 2)

    cc_start = (batch_mask_info[0, 0] == 1.0)
    cc_end   = (batch_mask_info[0, 1] == 1.0)

    ssim_list : list[float] = []
    psnr_list : list[float] = []

    # compute masks based on start
    if cc_start:
        # start index 
        start_offset = int(batch_mask_info[1, 0])
        
        # crop the real image
        t_real = real_b[batch_index, :, :start_offset, :]

        # crop the fake image
        t_fake = fake_b[batch_index, :, :start_offset, :]

        ssim_x, psnr_x = __compute_metrics(t_real, t_fake)
        ssim_list.append(ssim_x)
        psnr_list.append(psnr_x)

    if cc_end:
        backwards_index = int(batch_mask_info[1, 1])

        # crop the real image
        t_real = real_b[batch_index, :, backwards_index:, :]

        # crop the fake image
        t_fake = fake_b[batch_index, :, backwards_index:, :]

        ssim_x, psnr_x = __compute_metrics(t_real, t_fake)
        ssim_list.append(ssim_x)
        psnr_list.append(psnr_x)
    
    ssim_mean = torch.mean(torch.tensor(ssim_list)).item()
    psnr_mean = torch.mean(torch.tensor(psnr_list)).item()

    xdict = {
        'peak_signal_noise_ratio' : psnr_mean,
        'structural_similarity_index_measure' : ssim_mean
    }
    return xdict