def render_curved(self, font, word_text):
    """
    use curved baseline for rendering word
    """
    wl = len(word_text)
    isword = len(word_text.split()) == 1
    # do curved iff, the length of the word <= 10
    if not isword or wl > 10 or np.random.rand() > self.p_curved:
        return self.render_multiline(font, word_text)
    # create the surface:
    lspace = font.get_sized_height() + 1
    lbound = font.get_rect(word_text)
    fsize = (round(2.0 * lbound.width), round(3 * lspace))
    surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)
    # baseline state
    mid_idx = wl // 2
    BS = self.baselinestate.get_sample()
    curve = [BS['curve'](i - mid_idx) for i in xrange(wl)]
    curve[mid_idx] = -np.sum(curve) / (wl - 1)
    rots = [-int(math.degrees(math.atan(BS['diff'](i - mid_idx) / (font.size / 2)))) for i in xrange(wl)]
    bbs = []
    # place middle char
    rect = font.get_rect(word_text[mid_idx])
    rect.centerx = surf.get_rect().centerx
    rect.centery = surf.get_rect().centery + rect.height
    rect.centery += curve[mid_idx]
    ch_bounds = font.render_to(surf, rect, word_text[mid_idx], rotation=rots[mid_idx])
    ch_bounds.x = rect.x + ch_bounds.x
    ch_bounds.y = rect.y - ch_bounds.y
    mid_ch_bb = np.array(ch_bounds)
    # render chars to the left and right:
    last_rect = rect
    ch_idx = []
    for i in xrange(wl):
        # skip the middle character
        if i == mid_idx:
            bbs.append(mid_ch_bb)
            ch_idx.append(i)
            continue
        if i < mid_idx:  # left-chars
            i = mid_idx - 1 - i
        elif i == mid_idx + 1:  # right-chars begin
            last_rect = rect
        ch_idx.append(i)
        ch = word_text[i]
        newrect = font.get_rect(ch)
        newrect.y = last_rect.y
        if i > mid_idx:
            newrect.topleft = (last_rect.topright[0] + 2, newrect.topleft[1])
        else:
            newrect.topright = (last_rect.topleft[0] - 2, newrect.topleft[1])
        newrect.centery = max(newrect.height, min(fsize[1] - newrect.height, newrect.centery + curve[i]))
        try:
            bbrect = font.render_to(surf, newrect, ch, rotation=rots[i])
        except ValueError:
            bbrect = font.render_to(surf, newrect, ch)
        bbrect.x = newrect.x + bbrect.x
        bbrect.y = newrect.y - bbrect.y
        bbs.append(np.array(bbrect))
        last_rect = newrect
    # correct the bounding-box order:
    bbs_sequence_order = [None for i in ch_idx]
    for idx, i in enumerate(ch_idx):
        bbs_sequence_order[i] = bbs[idx]
    bbs = bbs_sequence_order
    # get the union of characters for cropping:
    r0 = pygame.Rect(bbs[0])
    rect_union = r0.unionall(bbs)
    # crop the surface to fit the text:
    bbs = np.array(bbs)
    '''
    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    '''
    # surf_arr, bbs = crop_safe(pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=5)
    surf_arr = pygame.surfarray.pixels_alpha(surf)
    '''
    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    '''
    surf_arr = surf_arr.swapaxes(0, 1)
    return surf_arr, word_text, bbs