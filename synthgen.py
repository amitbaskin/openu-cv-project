def render_text(self, rgb, imname, depth, seg, area, label, ninstance=1, viz=False):
    """
    rgb   : HxWx3 image rgb values (uint8)
    depth : HxW depth values (float)
    seg   : HxW segmentation region masks
    area  : number of pixels in each region
    label : region labels == unique(seg) / {0}
           i.e., indices of pixels in SEG which
           constitute a region mask
    ninstance : no of times image should be
                used to place text.

    @return:
        res : a list of dictionaries, one for each of
              the image instances.
              Each dictionary has the following structure:
                  'img' : rgb-image with text on it.
                  'bb'  : 2x4xn matrix of bounding-boxes
                          for each character in the image.
                  'txt' : a list of strings.

              The correspondence b/w bb and txt is that
              i-th non-space white-character in txt is at bb[:,:,i].

        If there's an error in pre-text placement, for e.g. if there's
        no suitable region for text placement, an empty list is returned.
    """
    try:
        # depth -> xyz
        xyz = su.DepthCamera.depth2xyz(depth)

        # find text-regions:
        regions = TextRegions.get_regions(xyz, seg, area, label)

        # find the placement mask and homographies:
        regions = self.filter_for_placement(xyz, seg, regions)

        # finally place some text:
        nregions = len(regions['place_mask'])
        if nregions < 1:  # no good region to place text on
            return []
    except:
        # failure in pre-text placement
        # import traceback
        traceback.print_exc()
        return []

    res = []
    for i in xrange(ninstance):
        place_masks = copy.deepcopy(regions['place_mask'])

        print
        colorize(Color.CYAN, " ** instance # : %d" % i)

        idict = {'img': [], 'charBB': None, 'wordBB': None, 'txt': None, 'font': None}

        m = self.get_num_text_regions(nregions)  # np.arange(nregions)#min(nregions, 5*ninstance*self.max_text_regions))
        reg_idx = np.arange(min(2 * m, nregions))
        np.random.shuffle(reg_idx)
        reg_idx = reg_idx[:m]

        placed = False
        img = rgb.copy()
        fonts = []
        itext = []
        ibb = []

        # process regions:
        num_txt_regions = len(reg_idx)
        NUM_REP = 5  # re-use each region three times:
        reg_range = np.arange(NUM_REP * num_txt_regions) % num_txt_regions
        for idx in reg_range:
            ireg = reg_idx[idx]
            try:
                if self.max_time is None:
                    txt_render_res = self.place_text(img, imname, place_masks[ireg],
                                                     regions['homography'][ireg],
                                                     regions['homography_inv'][ireg])
                else:
                    with time_limit(self.max_time):
                        txt_render_res = self.place_text(img, imname, place_masks[ireg],
                                                         regions['homography'][ireg],
                                                         regions['homography_inv'][ireg])
            except SystemExit:
                sys.exit()
            except TimeoutException, msg:
                print
                msg
                continue
            except:
                traceback.print_exc()
                # some error in placing text on the region
                continue

            if txt_render_res is not None:
                placed = True
                img, text, bb, collision_mask, font = txt_render_res
                font = font.__str__().split('/')[-1].rstrip(".ttf')")
                # update the region collision mask:
                place_masks[ireg] = collision_mask
                # store the result:
                fonts.append(font)
                itext.append(text)
                ibb.append(bb)

        if placed:
            # at least 1 word was placed in this instance:
            idict['img'] = img
            idict['txt'] = itext
            idict['charBB'] = np.concatenate(ibb, axis=2)
            idict['wordBB'] = self.char2wordBB(idict['charBB'].copy(), ' '.join(itext))
            idict['font'] = fonts
            res.append(idict.copy())

            if viz:
                viz_textbb(1, img, [idict['wordBB']], alpha=1.0)
                viz_masks(2, img, seg, depth, regions['label'])
                # viz_regions(rgb.copy(),xyz,seg,regions['coeff'],regions['label'])
                if i < ninstance - 1:
                    raw_input(colorize(Color.BLUE, 'continue?', True))

    return res


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
