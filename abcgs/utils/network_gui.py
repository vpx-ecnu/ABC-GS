import torch
from gs.gaussian_renderer import network_gui, render

def handle_network_gui(trainer):
    if network_gui.conn is None:
        network_gui.try_connect()
    while network_gui.conn is not None:
        try:
            net_image_bytes = None
            (
                custom_cam,
                do_training,
                trainer.config.pipe.convert_SHs_python,
                trainer.config.pipe.compute_cov3D_python,
                keep_alive,
                scaling_modifer,
            ) = network_gui.receive()
            if custom_cam is not None:
                net_image = render(
                    custom_cam, trainer.gaussians, trainer.config.pipe, trainer.background, scaling_modifer
                )["render"]
                net_image_bytes = memoryview(
                    (torch.clamp(net_image, min=0, max=1.0) * 255)
                    .byte()
                    .permute(1, 2, 0)
                    .contiguous()
                    .cpu()
                    .numpy()
                )
            network_gui.send(net_image_bytes, trainer.config.model.source_path)
            if do_training and (
                (trainer.iteration < int(trainer.total_iterations)) or not keep_alive
            ):
                break
        except Exception:
            network_gui.conn = None