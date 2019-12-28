# Copy-Pasting GAN
A TensorFlow implementation of paper **Object Discovery with a Copy-Pasting GAN**.

## Requirements
- Python 3.7
- TensorFlow 1.13.1

## Run
1. Prepare dataset
   1. Prepare foreground and background images(`jpg` format only);
   2. Put foreground images into `data/plane_sky/plane`;
   3. Put background images into `data/plane_sky/sky`;
   4. Resize foreground and background images to $240\times240$;(You can use `img_utils.py` in `utils` folder);

   - You can change image format in `train.py` and `utils/img_utils.py`;
   - You can change image folder in `cfgs.py`;
2. Train with `python main.py`.
3. You can find checkpoints and saved images in `logs`. :)

## Results
| ![foreground](imgs/foreground_img_0.jpg) | ![background](imgs/background_img_0.jpg) | ![shuffled_foreground](imgs/shuffled_foreground_img_0.jpg) |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                          Foreground                          |                          Background                          |                  Random selected foreground                  |
| ![foreground](imgs/random_mask_img_0.jpg) | ![background](imgs/grounded_fake_img_0.jpg) | ![shuffled_foreground](imgs/d_mask_grounded_fake_img_0.jpg) |
|                    Random generated Mask                     |             Grounded Fake<br>(with random mask)              |         Predicted mask from D<br>(for Grounded Fake)         |
| ![foreground](imgs/g_mask_foreground_img_0.jpg) | ![background](imgs/anti_shortcut_img_0.jpg) | ![shuffled_foreground](imgs/d_mask_anti_shortcut_img_0.jpg) |
|          Predicted mask from G<br>(for Foreground)           |       Anti-shortcut<br>(foreground+random foreground)        |         Predicted mask from D<br>(for Anti-shortcut)         |
| ![foreground](imgs/d_mask_foreground_img_0.jpg) | ![background](imgs/composited_img_0.jpg) | ![shuffled_foreground](imgs/d_mask_composite_img_0.jpg) |
|          Predicted mask from D<br>(for Foreground)           |       **Composited image**<br>(foreground+background)        |       Predicted mask from D<br>(for Composited image)        |
