def place_text(self, rgb, imname, collision_mask, H, Hinv):
    font = self.text_renderer.font_state.sample()
    font = self.text_renderer.font_state.init_font(font)
    render_res = self.text_renderer.render_sample(font, collision_mask)
    if render_res is None:  # rendering not successful
        return  # None
    else:
        text_mask, loc, bb, text = render_res
    # update the collision mask with text:
    collision_mask += (255 * (text_mask > 0)).astype('uint8')
    # warp the object mask back onto the image:
    text_mask_orig = text_mask.copy()
    bb_orig = bb.copy()
    text_mask = self.warpHomography(text_mask, H, rgb.shape[:2][::-1])
    bb = self.homographyBB(bb, Hinv)
    if not self.bb_filter(bb_orig, bb, text):
        # warn("bad charBB statistics")
        return  # None
    # get the minimum height of the character-BB:
    min_h = self.get_min_h(bb, text)
    # feathering:
    text_mask = self.feather(text_mask, min_h)
    im_final = self.colorizer.color(rgb, [text_mask], np.array([min_h]))
    '''
    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    '''
    path_to_save_crops = None
    font = str(font).split('/')[-1].rstrip(".ttf')")
    label = FontsDict.get_dict()[font]
    x, y, w, h = cv2.boundingRect(text_mask)
    crop = im_final[y: y + h, x: x + w]
    crop = cv2.resize(crop, (224, 224))
    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    title = imname.split('.')[-2]
    path = f'{path_to_save_crops}/{label}/{title}_{x}_{y}.jpg'
    cv2.imwrite(path, crop)
    '''
    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    '''
    return im_final, text, bb, collision_mask, font
