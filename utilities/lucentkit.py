# from lucent.optvis import render
# def print_MM(model: nn.Module):
#     model.eval()
#     for lay in get_model_layers(model):
#         print(lay)
# def visualizeChannel(model, channel_name, n_channel, path="./pic"):
#     try:
#         os.mkdir(path, 0o755)
#     except:
#         pass
#     model.eval()
#     for i in range(n_channel):
#         tmp = torch.tensor(render.render_vis(
#             model, f"{channel_name}:{i}"))[0][0]
#         tmp = tmp.permute(2, 0, 1)
#         save_image(tmp, f'{path}/{i}.png')