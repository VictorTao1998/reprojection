import torch
import torch.nn as nn
import torch.nn.functional as F


def local_contrast_norm(image, kernel_size=9, eps=1e-5):
    """compute local contrast normalization
    input:
        image: torch.tensor (batch_size, 1, height, width)
    output:
        normed_image
    """
    assert (kernel_size % 2 == 1), "Kernel size should be odd"
    batch_size, channel, height, width = image.shape
    assert (channel == 1), "Only support single channel image for now"
    unfold = nn.Unfold(kernel_size, padding=(kernel_size - 1) // 2)
    unfold_image = unfold(image)  # (batch, kernel_size*kernel_size, height*width)
    avg = torch.mean(unfold_image, dim=1).contiguous().view(batch_size, 1, height, width)
    std = torch.std(unfold_image, dim=1, unbiased=False).contiguous().view(batch_size, 1, height, width)

    normed_image = (image - avg) / (std + eps)

    return normed_image, std


class Fetch_Module(nn.Module):
    def __init__(self, padding_mode="zeros"):
        super(Fetch_Module, self).__init__()
        self.padding_mode = padding_mode

    def forward(self, right_img, disp):
        assert disp.shape == right_img.shape
        batch_size, channel, height, width = right_img.shape

        x_grid = torch.linspace(0., width - 1, width, dtype=disp.dtype, device=disp.device) \
            .view(1, 1, width, 1).expand((batch_size, height, width, 1))
        y_grid = torch.linspace(0., height - 1, height, dtype=disp.dtype, device=disp.device) \
            .view(1, height, 1, 1).expand((batch_size, height, width, 1))

        x_grid = x_grid - disp.permute(0, 2, 3, 1)
        x_grid = (x_grid - (width - 1) / 2) / (width - 1) * 2
        y_grid = (y_grid - (height - 1) / 2) / (height - 1) * 2
        xy_grid = torch.cat([x_grid, y_grid], dim=-1)

        reproj_img = F.grid_sample(right_img, xy_grid, padding_mode=self.padding_mode)

        return reproj_img


class Windowed_Matching_Loss(nn.Module):
    def __init__(self, lcn_kernel_size=9, window_size=33, sigma_weight=2,
                 invalid_reg_weight=1.0, invalid_weight=1.0):
        super(Windowed_Matching_Loss, self).__init__()
        self.lcn_kernel_size = lcn_kernel_size
        self.window_size = window_size
        self.sigma_weight = sigma_weight
        self.invalid_reg_weight = invalid_reg_weight
        self.invalid_weight = invalid_weight
        self.fetch_module = Fetch_Module()

    def forward(self, preds, data_batch):
        left_ir, right_ir = data_batch["img_L"], data_batch["img_R"]
        batch_size, channel, height, width = left_ir.shape
        assert (channel == 1)
        lcn_left_ir, left_std = local_contrast_norm(left_ir, self.lcn_kernel_size)
        lcn_right_ir, _ = local_contrast_norm(right_ir, self.lcn_kernel_size)

        disp = preds["refined_disp"]
        reproj_ir = self.fetch_module(lcn_right_ir, disp)
        reconstruct_err = F.l1_loss(lcn_left_ir, reproj_ir, reduction='none')
        C = left_std * reconstruct_err
        if self.window_size != 1:
            assert (self.window_size % 2 == 1)
            unfold = nn.Unfold(self.window_size, padding=(self.window_size - 1) // 2)
            C = unfold(C)  # (batch_size, window_size*window_size, height*width)
            Ixy = unfold(lcn_left_ir)
            wxy = torch.abs(lcn_left_ir.view(batch_size, 1, -1) - Ixy)
            wxy = torch.exp(-wxy / self.sigma_weight)
            C = (C * wxy).sum(1) / (wxy.sum(1))

        losses = {}
        # losses["C"] = C
        if "invalid_mask" in preds.keys() and "right_disp" in preds.keys():
            invalid_mask = preds["invalid_mask"]
            invalid_reg_loss = (- torch.log(1 - invalid_mask)).mean()
            losses["invalid_reg_loss"] = invalid_reg_loss * self.invalid_reg_weight
            C = C.view(left_std.shape)
            rec_loss = (C * (1 - invalid_mask)).mean()
            # rec_loss = C.mean()
            losses["rec_loss"] = rec_loss

            right_disp = preds["right_disp"]
            reproj_disp = self.fetch_module(right_disp, disp)

            disp_consistency = torch.abs(disp - reproj_disp)
            invalid_mask_from_consistency = (disp_consistency > 1).float()
            invalid_loss = (-torch.log(invalid_mask) * invalid_mask_from_consistency
                            - torch.log(1 - invalid_mask) * (1 - invalid_mask_from_consistency)).mean()
            losses["invalid_loss"] = invalid_loss * self.invalid_weight
        else:
            losses["rec_loss"] = C.mean()

        return losses


class Supervision_Loss(nn.Module):
    def __init__(self, invalid_weight=1.0):
        super(Supervision_Loss, self).__init__()
        self.invalid_weight = invalid_weight

    def forward(self, preds, data_batch):
        coarse_disp_pred = preds["coarse_disp"]
        refined_disp_pred = preds["refined_disp"]
        disp_gt = data_batch["disp_map"]
        coarse_disp_gt = F.interpolate(disp_gt, (coarse_disp_pred.shape[2], coarse_disp_pred.shape[3]))

        invalid_mask_pred = torch.clamp(preds["invalid_mask"], 1e-7, 1 - 1e-7)
        invalid_mask_gt = data_batch["invalid_mask"]
        valid_mask_gt = 1 - invalid_mask_gt
        coarse_valid_mask_gt = F.interpolate(valid_mask_gt, (coarse_disp_pred.shape[2], coarse_disp_pred.shape[3]))

        invalid_loss = -(torch.log(invalid_mask_pred) * invalid_mask_gt + torch.log(1 - invalid_mask_pred) * (
                1 - invalid_mask_gt)).mean()

        refined_disp_loss = (torch.abs(refined_disp_pred - disp_gt) * valid_mask_gt).sum() / (
                valid_mask_gt.sum() + 1e-7)
        coarse_disp_loss = (torch.abs(coarse_disp_pred - coarse_disp_gt) * coarse_valid_mask_gt).sum() / (
                coarse_valid_mask_gt.sum() + 1e-7)

        return {
            "coarse_disp_loss": coarse_disp_loss,
            "refined_disp_loss": refined_disp_loss,
            "invalid_loss": invalid_loss * self.invalid_weight,
        }


def test_lcn(image_path):
    import cv2
    source_image = cv2.imread(image_path, 0)
    source_image_tensor = torch.tensor(source_image).float().unsqueeze(0).unsqueeze(0)
    normed_image = local_contrast_norm(source_image_tensor)
    normed_image = normed_image.squeeze().numpy()
    import matplotlib.pyplot as plt
    plt.imshow(normed_image)
    plt.show()


def test_fetch():
    batch_size = 2
    height, width = 240, 320
    disp = 10

    left_image = torch.rand((batch_size, 1, height, width)).float()
    pad = torch.zeros((batch_size, 1, height, disp)).float()
    disp_map = torch.ones((batch_size, 1, height, width + disp)).float() * disp

    right_image = torch.cat([left_image, pad], dim=-1)

    fetch_module = Fetch_Module()
    reproj_image = fetch_module(right_image, disp_map)[:, :, :, disp:]

    print(torch.allclose(left_image, reproj_image, atol=1e-4))


if __name__ == '__main__':
    # test_lcn("/media/rayc/文档/我的坚果云/SU Lab/sofa_0/coded_light/0.png")
    # test_fetch()
    import numpy as np
    from tqdm import tqdm
    from activestereonet.data_loader.sulab_indoor_active import SuLabIndoorActiveSet

    max_disp = 136
    dataset = SuLabIndoorActiveSet(
        root_dir="/home/xulab/Nautilus/",
        mode="train",
        max_disp=max_disp,
        view_list_file="/home/xulab/Nautilus/example.txt"
    )

    # dataset = SuLabIndoorActiveSet(
    #     root_dir="/home/rayc",
    #     mode="train",
    #     max_disp=max_disp,
    #     view_list_file="/home/rayc/sulab_active/example.txt"
    # )

    data_batch = {}
    for k, v in dataset[1].items():
        if isinstance(v, torch.Tensor):
            data_batch[k] = v.unsqueeze(0).cuda()
        else:
            data_batch[k] = v
    loss_func = Windowed_Matching_Loss(window_size=25)
    preds = {"refined_disp": data_batch["disp_map"]}
    loss_dict = loss_func(preds=preds, data_batch=data_batch)
    # for k, v in loss_dict.items():
    #     print(k, v)

    # visualize matching cost along one row
    gt_disp = data_batch["disp_map"][0, 0].cpu().numpy()
    height, width = gt_disp.shape

    import matplotlib.pyplot as plt

    left_ir, right_ir = data_batch["left_ir"], data_batch["right_ir"]
    lcn_left_ir, left_std = local_contrast_norm(left_ir, loss_func.lcn_kernel_size)
    lcn_right_ir, _ = local_contrast_norm(right_ir, loss_func.lcn_kernel_size)
    lcn_left_ir = lcn_left_ir.cpu().numpy()[0, 0]
    lcn_right_ir = lcn_right_ir.cpu().numpy()[0, 0]

    C_dict = {}
    C_dict["gt"] = loss_dict["C"].cpu().numpy().reshape((height, width))
    disp_array = []
    for disp in tqdm(np.arange(0, max_disp, 2)):
        disp = float(disp)
        disp_array.append(disp)
        preds = {"refined_disp": torch.ones((1, 1, height, width)).float().cuda() * disp}
        loss_dict = loss_func(preds=preds, data_batch=data_batch)
        C_dict[disp] = loss_dict["C"].cpu().numpy().reshape((height, width))

    # for k, v in C_dict.items():
    #     print(k, v.shape)
    disp_array = np.array(disp_array)
    chosen_row_idx = (100, 200, 300, 400, 500)
    chosen_col = 400

    fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True, figsize=(20, 16))
    gt_disp_vis = gt_disp.copy()
    for row_idx in chosen_row_idx:
        gt_disp_vis[row_idx, :] = 300
    gt_disp_vis[:, chosen_col] = 300
    ax = axs[0, 0]
    ax.imshow(gt_disp_vis)
    ax.scatter((chosen_col,)*len(chosen_row_idx), chosen_row_idx, marker='^', c='r', s=5)
    ax.set_title("GT disp")
    # plt.colorbar()
    # plt.show()

    ax = axs[0, 1]
    ax.set_title("LCN Left IR")
    ax.imshow(lcn_left_ir)
    ax.scatter((chosen_col,)*len(chosen_row_idx), chosen_row_idx, marker='^', c='r', s=5)


    ax = axs[1, 1]
    gt_disp_dict = {}
    min_disp_dict = {}
    for row_idx in chosen_row_idx:
        matching_cost = []
        for disp in disp_array:
            matching_cost.append(C_dict[disp][row_idx, chosen_col])
        matching_cost = np.array(matching_cost)
        ax.plot(disp_array, matching_cost, label=f"{row_idx}")
        gt_disp_dict[row_idx] = gt_disp[row_idx, chosen_col]
        min_disp_dict[row_idx] = disp_array[matching_cost.argmin()]

    ax = axs[1, 0]
    ax.set_title("LCN right IR")
    ax.imshow(lcn_right_ir)
    for row_idx in chosen_row_idx:
        ax.scatter(chosen_col-gt_disp_dict[row_idx], row_idx, marker='*', c='r', s=10)
        ax.scatter(chosen_col-min_disp_dict[row_idx], row_idx, marker='s', c='b', s=5)

    for k, v in gt_disp_dict.items():
        print(k, "gt: ", v, "min: ", min_disp_dict[k])

    plt.legend()
    plt.show()

    # gt_disp_loss = {}
    # rand_disp_loss = {}
    #
    # for window_size in tqdm(np.arange(5, 27, 2)):
    #     window_size = int(window_size)
    #     loss_func = Windowed_Matching_Loss(window_size=window_size)
    #     preds = {"refined_disp": data_batch["disp_map"]}
    #     loss_dict = loss_func(preds=preds, data_batch=data_batch)
    #     gt_disp_loss[window_size] = loss_dict["rec_loss"].cpu().numpy()
    #     preds = {"refined_disp": torch.rand((1, 1, 600, 800)).float().cuda() * max_disp}
    #     loss_dict = loss_func(preds=preds, data_batch=data_batch)
    #     rand_disp_loss[window_size] = loss_dict["rec_loss"].cpu().numpy()
    #
    # for k in sorted(gt_disp_loss.keys()):
    #     print(k, "gt: {:.4f}, rand: {:.4f}".format(gt_disp_loss[k], rand_disp_loss[k]))

