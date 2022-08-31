import numpy as np
import math
from collections import defaultdict
import h5py

def freq_to_cents_np(freq, cents_mapping, std=25):
    frequency_reference = 10
    c_true = 1200 * np.log2((np.array(freq)+1e-5) / frequency_reference)
    c_true = np.expand_dims(c_true, 1)
    cents_mapping = np.tile(np.expand_dims(cents_mapping,0), [c_true.shape[0],1])
    target = np.exp(-(cents_mapping - c_true) ** 2 / (2 * std ** 2))
    pitch_cent = np.sum(target.reshape([c_true.shape[0], 6, 120]), 1)
    return pitch_cent

def freq_to_cents(freq, cents_mapping, std=25):
    frequency_reference = 10
    c_true = 1200 * math.log((freq+1e-5) / frequency_reference,2)
    target = np.exp(-(cents_mapping - c_true) ** 2 / (2 * std ** 2))
    pitch_cent = np.sum(target.reshape([6, 120]), 0)
    return pitch_cent

def get_pitchvalues(pitches_arr):
    cents_mapping = np.linspace(0, 7190, 720) + 2051.1487628680297
    return freq_to_cents_np(pitches_arr, cents_mapping)

def reorder_tonic(pitchvalue_prob, tonic_freq):
    cents_mapping = np.linspace(0, 7190, 720) + 2051.1487628680297
    tonic_pv_arr = freq_to_cents(tonic_freq, cents_mapping)
    tonic_pv = np.argmax(tonic_pv_arr)
    return np.roll(pitchvalue_prob, -tonic_pv, axis=1)

def normalize(z):
    z_min = np.min(z)
    return (z - z_min)/(np.max(z)-z_min+1e-6)

def compare(a,b,x,asc):
    if not asc:
        a, b = b, a
    if a <= modulo_add(a,x) <= modulo_add(a,b):
        return True
    return False

def modulo(x):
    return x%120

def modulo_add(x,y):
    mx = modulo(x)
    my = modulo(y)
    if mx>my:
        return my+120
    return my

def relax_fun(p,add,r=4):
    if add:
        return modulo(p+4)
    return modulo(p-4)


def get_dist_btw_idx(pitchvalue_prob, start_idx, end_idx):
    dist = 0
    for i in range(start_idx, end_idx + 1):
        dist += pitchvalue_prob[i]
    return dist


def get_dist_btw_shortlisted_idxs(pitchvalue_prob, shortlisted_idxs, off_start=0, off_end=None):
    if off_end is None:
        off_end = len(pitchvalue_prob) - 1
    dist = 0
    for sidx in shortlisted_idxs:
        i1, i2, i3, i4 = sidx[0], sidx[1], sidx[2], sidx[3]
        if i2 < off_start:
            continue
        if i3 > off_end:
            continue
        if i1 <= off_start <= i2:
            i1 = off_start
        if i3 <= off_end <= i4:
            i4 = off_end

        dist += get_dist_btw_idx(pitchvalue_prob, i1, i4)
    return normalize(dist)


def update_shortlisted_index(shortlisted_index, pitch_st_mapping, start_index, end_index):
    psm_ss = pitch_st_mapping[start_index][0]
    psm_se = pitch_st_mapping[start_index][1]
    psm_es = pitch_st_mapping[end_index][0]
    psm_ee = pitch_st_mapping[end_index][1]

    shortlisted_index.append((psm_ss, psm_se, psm_es, psm_ee))


def compute_spd_ps_pe(start_idx, pitches_arg, pitch_st_mapping, ps, pe, asc, relax=4):
    #     width = None
    #     prev_dist = 0
    n = len(pitches_arg)
    k = 0
    start = True
    end = False
    b = 0
    start_id = 0
    end_id = 0
    si = 0
    prev_s = None
    start_index = -1
    end_index = -1
    idx = start_idx[si]
    dist_pres = False
    dist_added = False
    #     for idx in range(start_idx[si], n):
    #     base_key = '{}_{}_{}_{}'
    shortlisted_index = []
    while idx < n:

        #         if si>=len(start_idx):
        #             break
        #         if si>=len(start_idx):
        #             break
        #         if idx>=start_idx[si]:
        #             si+=1
        p = pitches_arg[idx]

        if start and end and p != pe:
            update_shortlisted_index(shortlisted_index, pitch_st_mapping, start_index, end_index)
            # This verfies Equation 13; Page 4
            #             add_lm_file(lm_file, pitch_st_mapping, start_index, end_index, prev_dist, base_key)
            #             lm_file[base_key.format()]
            #             shortlisted_index.append()
            #             prev_dist = 0
            start = False
            end = False
            start_index = -1
            end_index = -1
            dist_pres = True

        if start and p == pe:
            end = True  # This verifies Equation 14; Page 4
            end_index = idx
        #             prev_dist += get_dist_btw_idx(pitchvalue_prob, pitch_sst_mapping, idx)

        #         if start and (compare(ps, pe, p, asc)) and (not end): # This verifies Equation 15; Page 4
        #             prev_dist += get_dist_btw_idx(pitchvalue_prob, pitch_st_mapping, idx)
        #           prev_dist_start.append(idx)

        if p == ps:
            start = True  # This verifies Equation 12; Page 4
            start_index = idx
        #             prev_dist += get_dist_btw_idx(pitchvalue_prob, pitch_st_mapping, idx)

        if not (compare(ps, pe, p, asc)):
            #             prev_dist = 0
            start = False
            end = False
            start_index = -1
            end_index = -1

        if p == ps:
            si += 1
        if not start:
            if si >= len(start_idx):
                break
            else:
                idx = start_idx[si]
                idx -= 1
        idx += 1

    # This handles an edge case where prev_dist is not empty but not yet been added to cum_pitch_dist
    if start and end:
        update_shortlisted_index(shortlisted_index, pitch_st_mapping, start_index, end_index)
    #         add_lm_file(lm_file, pitch_st_mapping, start_index, end_index, prev_dist, base_key)

    if not dist_pres:
        #         prev_dist = get_pd_between_pspe(pitchvalue_prob, ps, pe, asc)  # Return simple pitch distributin incase SPD is empty lines 254, 255
        #         add_lm_file(lm_file, pitch_st_mapping, start_index, end_index, prev_dist, base_key)
        update_shortlisted_index(shortlisted_index, pitch_st_mapping, 0, n - 1)
    return shortlisted_index


def get_all_smooth_pitch_values(std=25):
    c_note = freq_to_cents(32.7 * 2, std)
    all_notes = np.zeros([120, 120])
    for p in range(120):
        all_notes[p] = get_smooth_pitch_value(c_note, p)

    return all_notes, c_note


def get_smooth_pitch_value(c_note, note):
    return np.roll(c_note, note, axis=-1)


def gauss_smooth(raga_feat):
    all_notes, c_note = get_all_smooth_pitch_values(std=25)
    smooth = np.zeros([12, 12, 120, 2])
    for i in range(12):
        for j in range(12):
            if i == j:
                continue
            for k in range(0, 2):
                smooth[i, j, :, k] = gauss_smooth_util(raga_feat[i, j, :, k], all_notes)
    return smooth


def gauss_smooth_util(arr1, all_notes):
    smooth = 0
    for i in range(120):
        smooth = smooth + all_notes[i] * arr1[i]

    #     smooth = np.power(normalize(smooth), 0.8)
    smooth = normalize(smooth)
    return smooth


def get_std_idx(pitches_arg, relax=4):
    pitch_dict = defaultdict(list)
    std_pitches = []
    pitch_st_mapping = []
    prev_p = None
    k = 0
    for i, p in enumerate(pitches_arg):
        if prev_p is None:
            std_pitches.append(p // 10)
            pitch_dict[p // 10].append(k)
        elif prev_p // 10 != p // 10:
            k += 1
            std_pitches.append(p // 10)
            pitch_dict[p // 10].append(k)
        if k >= len(pitch_st_mapping):
            pitch_st_mapping.append([i, i])
        else:
            if pitch_st_mapping[-1][1] + 1 == i:
                pitch_st_mapping[-1][1] = i
            else:
                pitch_st_mapping.append([i, i])
        prev_p = p

    return pitch_dict, std_pitches, pitch_st_mapping

def full_spd(pitches_arg, lm_file):
    pitch_dict, std_pitches, pitch_st_mapping = get_std_idx(pitches_arg)
    for asc in [True, False]:
        for s in range(0, 12, 1):
            start_idx = pitch_dict[s]
            for e in range(0, 12, 1):
                if s==e:
                    continue
#                 lm_file_group = lm_file.create_group("{}_{}_{}_{}".format(mbid, s, e, asc))
                shortlisted_index = compute_spd_ps_pe(start_idx, std_pitches, pitch_st_mapping, s, e, asc)
                lm_file["{}_{}_{}".format(s, e, asc)] = shortlisted_index

def generate_spd_idx_all_files(pitchvalue_prob):
    spd_idx_lm_file = {}
    pitches_arg = np.argmax(pitchvalue_prob, axis=1)
    #                 pitch_dict = get_full_spd_st(pitches_arg, mbid, spd_idx_lm_file)
    full_spd(pitches_arg, spd_idx_lm_file)
    return spd_idx_lm_file

def get_cliped_dist(s,e,asc,dist,clip=15):
    s10 = s*10
    e10 = e*10
    if asc:
        relax = clip
    else:
        relax = -clip
    i = modulo(s10-relax)
    j = modulo(e10+relax)
    m = 0
    while i!=j:
        if asc:
            i = modulo(i+1)
        else:
            i = modulo(i-1)
        m+=1
    dist_sliced = np.zeros(m)
    i = modulo(s10-relax)
    j = modulo(e10+relax)
    if (m<=abs(relax)):
        dist_sliced = dist
    else:
        m=0
        while i!=j:
            if asc:
                i = modulo(i+1)
            else:
                i = modulo(i-1)
            dist_sliced[m] = dist[i]
            m+=1
    return dist_sliced

def get_spd_from_idx(pitchvalue_prob, off_start=0, off_end=None):
    dist_hist = get_dist_btw_idx(pitchvalue_prob, 0, len(pitchvalue_prob)-1)
    dist_hist = normalize(dist_hist)
    spd_idx_lm_file = generate_spd_idx_all_files(pitchvalue_prob)
    full_spd_dist = np.zeros([12,12,120,2])
    for asc in [True, False]:
        asc_int = 1-int(asc)
        for s in range(0,12,1):
            for e in range(0, 12, 1):
                if s==e:
                    full_spd_dist[s,e,:,asc_int] = dist_hist
                    continue
                shortlisted_idxs = spd_idx_lm_file['{}_{}_{}'.format(s, e, asc)]
                dist = get_dist_btw_shortlisted_idxs(pitchvalue_prob,shortlisted_idxs, off_start, off_end)

                if np.sum(dist)==0:
                    full_spd_dist[s, e, :, asc_int] = dist_hist
                else:
                    full_spd_dist[s,e,:,asc_int] = dist
    return full_spd_dist, dist_hist

def generate_full_spd_cache(pitchvalue_prob):
    full_spd_dist, dist_hist = get_spd_from_idx(pitchvalue_prob)
    return full_spd_dist, dist_hist

def get_raga_feat_and_predict(knn_models, pitchvalue_prob, n_labels):
    full_spd_dist, dist_hist = generate_full_spd_cache(pitchvalue_prob)
    pred_proba = np.zeros([25, n_labels])
    for wd in range(0,250,10):
        spd_knn = knn_models[wd]
        n_rows = 1
        if wd == 0:
            feat = np.zeros([n_rows, 120])
            feat[0] = dist_hist
        elif 0<wd<120:
            feat = []
            feat_curr = []
            for s in range(0,120,10):
                e = modulo(s+wd)
                if s==e:
                    continue
                s10 = s//10
                e10 = e//10
                hist_1 = full_spd_dist[s10,e10,:,0]
                hist_2 = full_spd_dist[e10,s10,:,1]
                hist_1 = get_cliped_dist(s,e,True,hist_1,clip=15)
                hist_2 = get_cliped_dist(e,s,False,hist_2,clip=15)
                feat_curr.append(hist_1)
                feat_curr.append(hist_2)
#                 if feat_curr is None:
#                     feat_curr = np.concatenate([hist_1, hist_2], axis=-1)
#                 else:
#                     feat_curr = np.concatenate([feat_curr, hist_1], axis=-1)
#                     feat_curr = np.concatenate([feat_curr, hist_2], axis=-1)
            feat_curr = np.concatenate(feat_curr, axis=-1)
#             feat[row[0]] = feat_curr
            feat.append(feat_curr)
        elif 120<=wd<240:
            feat = []
            s = wd-120
            feat_curr = []
            for e in range(0,120,10):
                if s==e:
                    continue
                s10 = s//10
                e10 = e//10
                hist_1 = full_spd_dist[s10,e10,:,0]
                hist_2 = full_spd_dist[e10,s10,:,1]
                hist_1 = get_cliped_dist(s,e,True,hist_1,clip=15)
                hist_2 = get_cliped_dist(e,s,False,hist_2,clip=15)
                feat_curr.append(hist_1)
                feat_curr.append(hist_2)
            feat_curr = np.concatenate(feat_curr, axis=-1)
            feat.append(feat_curr)
        else:
            feat = []
            hist_1 = np.array(full_spd_dist)
            feat.append(np.reshape(hist_1, [-1]))
        feat = np.array(feat)
        pred_proba[wd//10] = spd_knn.predict(feat)
    return pred_proba

def get_range_dict(relax_sign, asc):
    lim = 55
    from collections import defaultdict
    range_dict = defaultdict(list)
    inv_range_dict = defaultdict(list)

    for i in range(0, 60, 5):
        for p in range(60):
            for j in range(0,lim+1,5):
                if asc:
                    j = (i + j + 60) % 60
                else:
                    j = (60 - j + i) % 60
                if i==j:
                    continue
                if compare_notes(i - relax_sign,  j + relax_sign, p, asc):
                    range_dict[(i,p)].append(j)
                    inv_range_dict[j].append((i,p))
                    # range_dict.append(((p,i,j)))
                    # range_dict[(p,i)] = j
    return range_dict, inv_range_dict

if __name__ == '__main__':
    # pitches_arg = np.arange(0,60)
    # breaks = np.ones(60)
    # get_all_raga_features(pitches_arg, breaks)
    mbid = '99dcaebe-ab49-4fa3-ab6c-a9458143af8e'
    # raga_feat = get_raga_feat_cache(mbid, 0, 1998*5)
    print(raga_feat.shape)
    # range_dict, inv_range_dict = get_range_dict(0,True)
    # print(inv_range_dict)
