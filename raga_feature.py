import numpy as np
import math
from collections import defaultdict


def freq_to_cents_1(freq, std=25, reduce=False):
    frequency_reference = 10
    c_true = 1200 * math.log(freq / frequency_reference, 2)

    cents_mapping = np.linspace(0, 7180, 360) + 1997.3794084376191
    target = np.exp(-(cents_mapping - c_true) ** 2 / (2 * std ** 2))

    if reduce:
        return np.sum(target.reshape([6, 60]), 0)
    return target


def get_all_smooth_notes():
    c_note = freq_to_cents_1(31.7 * 2, reduce=True)
    all_notes = np.zeros([60, 60])
    for p in range(60):
        all_notes[p] = get_smooth_note(c_note, p)

    return all_notes, c_note

def get_smooth_note(c_note, note):
    return np.roll(c_note, note, axis=-1)

all_notes, c_note = get_all_smooth_notes()

def freq_to_cents(freq, std=25, reduce=False):
    frequency_reference = 10
    c_true = 1200 * math.log(freq / frequency_reference, 2)

    cents_mapping = np.linspace(0, 7180, 360) + 1997.3794084376191
    target = np.exp(-(cents_mapping - c_true) ** 2 / (2 * std ** 2))

    if reduce:
        return np.sum(target.reshape([6, 60]), 0)
    return target

def get_all_smooth_notes():
    c_note = freq_to_cents(31.7 * 2, reduce=True)
    all_notes = np.zeros([60, 60])
    for p in range(60):
        all_notes[p] = get_smooth_note(c_note, p)

    return all_notes, c_note


def get_raga_feature(pitches_unique, pitches_count, breaks_unique, func, c_note, all_notes, asc=True, transpose=False, relax=0):
    r = [0,1]
    lim = 55
    #     ran = np.random.randint(pitches.shape[0]-1998)
    #     ran = 63894
    # pitches_arg = np.argmax(pitches[ran:ran + 1998 * 5, :], axis=1)

    raga_feature = np.zeros([12, 12, 60])
    pitches_dist_dict = defaultdict(lambda: [[-1,-1,-1,-1,np.zeros(60)]])
    for a in range(*r):
        pitches_arg_r = (pitches_unique + a + 60) % 60
        m = c_note[a]
        for i in range(0, 60, 5):
            for j in range(0, lim + 1, 5):
                i = (i + 60) % 60

                if asc:
                    j = (i + j + 60) % 60
                else:
                    j = (60 - j + i) % 60

                #                 j=(j+60)%60
                if i == j:
                    continue
                s = i
                e = j
                #             print(s,e)
                durations = []
                dist_asc = 0

                #             pitches_r = np.roll(pitches, a, axis=1)
                #             pitches_r = pitches_r[:1998,:]
                temp = func(pitches_arg_r, pitches_count, pitches_dist_dict, s, e, asc, relax)
                #             temp = normalize(temp)
                #             temp = np.concatenate([temp,temp,temp], axis=-1)
                #             temp1 = np.zeros(lim+10)
                #             for k in range(lim+10):
                #                 if compare_notes((i-5+60)%60,(i+35+5)%60,(i+k+60-5)%60, asc=True, relax=0):
                #                     temp1[k]=temp[(i+k+60-5)%60]
                # #             temp = temp[i:i+35]
                raga_feature[i // 5, j // 5, :] += m * temp
    #                 raga_feature[i//5,j//5,:]+=m*np.sum(temp)
    #                 raga_feature[i//5,j//5,:]+=m*temp
    #             temp, dur = get_dist_btw_notes(pitches_arg,e,s, False, all_notes)
    #             dist_asc+=m*temp
    #             durations.extend(m*dur)
    #     raga_feature_red = compute_dot_prod(raga_feature)
    # raga_feature_red = get_n_peaks_raga_feat(raga_feature)
    # raga_feature_red = normalize(raga_feature)
    raga_feature_red = raga_feature
    if transpose:
        raga_feature_red = np.transpose(raga_feature_red, [1,0,2])

    # hist = normalize(np.mean(pitches,0))
    # for i in range(raga_feature_red.shape[0]):
    #     raga_feature_red[i,i,:] = hist

    # raga_feature_red = stadardize(raga_feature_red)

    raga_features_list = [[] for i in range(12)]
    # for i in range(0,12):
    #     raga_features_list.append(np.zeros([12,5*i]))

    # raga_features_list = np.concatenate(raga_features_list, axis=1) # 12,390


    for i in range(12):
        # temp = np.zeros([12, (i+1)*5])
        for j in range(12):
            if i!=j:
                # temp = get_span(raga_feature_red[i, j], i * 5, j * 5)
                temp = raga_feature_red[i, j]
                if i>j:
                    d = j+12-i
                else:
                    d = j-i
                raga_features_list[d].append(temp)

    #     plt.imshow(np.expand_dims(np.mean(raga_feature_red,2),2), cmap='hot', interpolation='nearest')
    # plt.imshow(np.expand_dims(raga_feature_red, 2), cmap='hot', interpolation='nearest')
    return np.concatenate([raga_features_list[1:12][i] for i in range(11)],-1)

def get_span(hist, s, e):
    if e<s:
        e=e+60
    hist_conc = np.concatenate([hist, hist], axis=-1)
    return hist_conc[s:e]

def stadardize(z):
    return (z - np.mean(z))/(np.std(z))

def normalize(z):
    z_min = np.min(z)
    return (z - z_min)/(np.max(z)-z_min+0.001)


def compare_notes(a, b, x, asc=True, relax=0):
    a = (a + 60) % 60
    b = (b + 60) % 60
    if not asc:
        a, b = b, a
    if a > b:
        if x <= b:
            x = x + 60
        b = b + 60
    if x >= a and x <= b:
        return True

    return False


def get_dist_btw_notes(pitches_arg, pitches_count, allowed_indices, note_a, note_b, asc, relax):

    if asc:
        relax_sign = relax
    else:
        relax_sign = -relax

    note_a = (60 + note_a) % 60
    note_b = (60 + note_b) % 60

    pitch_dist_cache = []

    consumed_indices = set()
    pitches_arg_len = len(pitches_arg)
    for ai in allowed_indices:
        if ai in consumed_indices:
            continue
        prev_dist = np.zeros(60)
        start = False
        end = False
        c = 0
        k = 0
        for idx in range(ai,pitches_arg_len):
            p = pitches_arg[idx]

            if compare_notes(note_a-relax_sign, note_a+relax_sign, p, asc,0):
                start = True
                consumed_indices.add(idx)

            if start and compare_notes(note_b-relax_sign, note_b+relax_sign, p, asc,0):
                end = True

            if start and (compare_notes(note_a-relax_sign, note_b+relax_sign, p, asc,0) or compare_notes(0-relax_sign, 0+relax_sign, p, asc,0)):
                prev_dist[p] = prev_dist[p] + pitches_count[idx]
                k=1

            if start and end and not (compare_notes(note_a-relax_sign, note_b+relax_sign, p, asc,0) or compare_notes(0-relax_sign, 0+relax_sign, p, asc,0)):
                if len(pitch_dist_cache)==0:
                    cum_pitch_dist = prev_dist
                else:
                    cum_pitch_dist = pitch_dist_cache[-1][1:] + prev_dist
                pitch_dist_cache.append(np.concatenate([[ai],cum_pitch_dist], axis=-1))
                c += 1
                k=0
                break

            if not (compare_notes(note_a-relax_sign, note_b+relax_sign, p, asc,0) or compare_notes(-relax_sign, relax_sign, p, asc,0)):
                k=0
                break

        if k==1:
            if len(pitch_dist_cache) == 0:
                cum_pitch_dist = prev_dist
            else:
                cum_pitch_dist = pitch_dist_cache[-1][1:] + prev_dist
            pitch_dist_cache.append(np.concatenate([[ai], cum_pitch_dist], axis=-1))

        # if c == 0 and k==0:
        #     pitch_dist += np.zeros(60)

    return pitch_dist_cache

def get_dist_btw_notes_1(pitches, pitches_count, pitch_hist_dic, note_a, note_b, asc=True, relax=0):
    # if asc:
    #     note_a = note_a-relax
    #     note_b = note_b+relax
    # else:
    #     note_a = note_a+relax
    #     note_b = note_b-relax

    note_a = (60 + note_a) % 60
    note_b = (60 + note_b) % 60
    #     ct_frames = (16000*cutoff - 1024)/480 + 1
    #     c_note = freq_to_cents(31.7*2, reduce=True)
    pitch_dist = 0
    prev_note = note_a
    prev_dist = np.zeros(60)
    #     pitches_arg = np.argmax(pitches, 1)
    pitches_arg = pitches
    start = False
    end = False
    c = 0
    k=0

    if asc:
        relax_sign = relax
    else:
        relax_sign = -relax
    #     durations = []
    start_p = -1
    end_p = -1
    for idx in range(len(pitches_arg)):
        p = pitches_arg[idx]

        if compare_notes(note_a-relax_sign, note_a+relax_sign, p, asc,0):
            if not start:
                start_p = idx
            start = True

        if start and compare_notes(note_b-relax_sign, note_b+relax_sign, p, asc,0):
            end = True


        if start and compare_notes(note_a-relax_sign, note_b+relax_sign, p, asc,0):
            prev_dist[p] = prev_dist[p] + pitches_count[idx]
            end_p = idx
            k=1

        if start and end and not compare_notes(note_a-relax_sign, note_b+relax_sign, p, asc,0):
            pitch_dist += prev_dist
            prev_dist = np.zeros(60)
            start = False
            end = False
            start_p = -1
            end_p = -1
            c += 1
            k=0

        if not compare_notes(note_a-relax_sign, note_b+relax_sign, p, asc,0):
            prev_dist = np.zeros(60)
            start = False
            end = False
            start_p = -1
            end_p = -1
            k=0

        # if note_a==10:
        #     if start and end:


    if k==1:
        pitch_dist += prev_dist
        # pitch_hist_dic[start_p].append([note_a, note_b, end_p, pitches_arg[end_p], prev_dist])

    if c == 0 and k==0:
        return np.zeros(60)
    return pitch_dist

def compare_pitch_dist(note_a, note_b, p, pitch_hist_dic):
    if p not in pitch_hist_dic:
        return None
    pitch_dist_cache = pitch_hist_dic[p]
    d = 0
    spdc = None

    for pdc in pitch_dist_cache:
        prev_note_a = pdc[0]
        prev_note_b = pdc[1]
        print(note_a, note_b, prev_note_a, prev_note_b)
        if compare_notes(note_a, note_b, prev_note_a, asc=True) and compare_notes(note_a, note_b, prev_note_b, asc=True):
            spdc = pdc
            break
            # d1 = (note_a - prev_note_a+60)%60
            # d2 = (note_b - prev_note_b + 60) % 60
            # td = d1+d2
            # if td<=d:
            #     d = td
            #     spdc = pdc

    return spdc


def get_prev_note_b_dist_dict(key, note_a, prev_note_b_dist_dict):
    while key[1]!=note_a:
        if key in prev_note_b_dist_dict:
            return prev_note_b_dist_dict[key]
        key = (key[0], (key[1]-5+60)%60)

    return np.zeros(60)

def get_fast_raga_feature(pitches_arg, pitches_count, asc, relax=0):

    lim = 55

    pitch_dist_list = {}


    note_a_dict = get_note_a_dict(relax, asc=asc)
    allowed_indices_dict = get_allowed_indices(pitches_arg, note_a_dict)


    for i in range(0, 60, 5):
        allowed_indices = allowed_indices_dict[i]
        for j in range(0, lim + 1, 5):
            i = (i + 60) % 60
            s = i

            if asc:
                e = (i + j + 60) % 60
            else:
                e = (60 - j + i) % 60

            if s==e:
                continue

            pitch_dist_cache_1 = get_dist_btw_notes(pitches_arg, pitches_count, allowed_indices, s, e, asc,
                                                    relax)
            pitch_dist_cache_2 = get_strict_dist_btw_notes(pitches_arg, pitches_count, allowed_indices, s, e, asc,
                                                           relax)

            pitch_dist_list[(s,e)] = [pitch_dist_cache_1, pitch_dist_cache_2]
            # pitch_dist_list[(s, e)] = temp
            # break

    return pitch_dist_list

    # print(pitch_note_b_dist_dict.keys())

def get_allowed_indices(pitches_arg, note_a_dict):

    allowed_indices = defaultdict(list)
    for idx in range(len(pitches_arg)):
        p = pitches_arg[idx]
        note_a = note_a_dict[p]
        if note_a!=-1:
            allowed_indices[note_a].append(idx)
    return allowed_indices


def get_note_a_dict(relax, asc):
    if asc:
        relax_sign = relax
    else:
        relax_sign = -relax

    note_b_dict = defaultdict(lambda: -1)
    for note in range(0,60,5):
        for p in range(60):
            if compare_notes(note - relax_sign, note + relax_sign, p, asc, 0):
                note_b_dict[p] = note
    return note_b_dict

def get_note_b_dict_1(note_a, note_b_list, relax_sign, asc):
    note_b_dict = defaultdict(lambda: -1)
    note_b_list = np.sort(np.array(note_b_list))


    for p in range(60):
        for note in note_b_list:
            if compare_notes(note_a - relax_sign, note + relax_sign, p, asc, 0):
                note_b_dict[p] = note
                break
    return note_b_dict

def get_seq_dict(pitches_arg,relax_sign, asc):
    note_b_dict = get_note_b_dict_2(relax_sign, asc)


    seq_dict = defaultdict(lambda:-1)
    for p in pitches_arg:
        note_b = note_b_dict[p]
        if note_b!=-1:
            seq_dict[p] = note_b

def get_strict_dist_btw_notes(pitches_arg, pitches_count, allowed_indices, note_a, note_b, asc, relax):

    if asc:
        relax_sign = relax
    else:
        relax_sign = -relax

    note_a = (60 + note_a) % 60
    note_b = (60 + note_b) % 60

    pitch_dist_cache = []

    consumed_indices = set()
    pitches_arg_len = len(pitches_arg)
    for ai in allowed_indices:
        if ai in consumed_indices:
            continue
        prev_dist = np.zeros(60)
        start = False
        end = False
        c = 0
        k = 0
        prev_max = note_a
        for idx in range(ai,pitches_arg_len):
            p = pitches_arg[idx]

            if compare_notes(note_a-relax_sign, note_a+relax_sign, p, asc,0):
                start = True
                consumed_indices.add(idx)
                prev_max = note_a

            if start and compare_notes(note_b-relax_sign, note_b+relax_sign, p, asc,0):
                end = True

            if start and (compare_notes(prev_max-relax_sign, note_b+relax_sign, p, asc,0) or compare_notes(0-relax_sign, 0+relax_sign, p, asc,0)):
                prev_dist[p] = prev_dist[p] + pitches_count[idx]
                k=1

            if start and end and not (compare_notes(prev_max-relax_sign, note_b+relax_sign, p, asc,0) or compare_notes(0-relax_sign, 0+relax_sign, p, asc,0)):
                if len(pitch_dist_cache)==0:
                    cum_pitch_dist = prev_dist
                else:
                    cum_pitch_dist = pitch_dist_cache[-1][1:] + prev_dist
                pitch_dist_cache.append(np.concatenate([[ai],cum_pitch_dist], axis=-1))
                c += 1
                k=0
                break

            if not (compare_notes(prev_max-relax_sign, note_b+relax_sign, p, asc,0) or compare_notes(-relax_sign, relax_sign, p, asc,0)):
                k=0
                break

            if compare_notes(prev_max, note_b + relax_sign, p, asc):
                prev_max = p

        if k==1:
            if len(pitch_dist_cache) == 0:
                cum_pitch_dist = prev_dist
            else:
                cum_pitch_dist = pitch_dist_cache[-1][1:] + prev_dist
            pitch_dist_cache.append(np.concatenate([[ai], cum_pitch_dist], axis=-1))

        # if c == 0 and k==0:
        #     pitch_dist += np.zeros(60)

    return pitch_dist_cache

def get_note_b(note_a, p, range_dict):

    note_b_range = range_dict[(note_a,p)]
    if len(note_b_range)==0:
        return None
    note_b = note_b_range[0]
    # if note_b==5:
    return note_b
    # note_b = (note_a+5+60)%60


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


def get_all_raga_features(pitches_arg):
    relax=2

    c_note, all_notes = get_all_smooth_notes()
    # print(len(pitches_unique), len(pitches_count))
    pitches_dist_dict = {}
    #
    # f2 = get_dist_btw_notes(pitches_unique, pitches_count, pitches_dist_dict, 0, 5, asc=True, relax=relax)
    # print(pitches_dist_dict.keys())

    # f2 = get_fast_raga_feature(pitches_unique, pitches_count, relax=relax)
    # f3 = get_fast_raga_feature(pitches_unique, pitches_count, relax=relax)
    # f4 = get_fast_raga_feature(pitches_unique, pitches_count, relax=relax)

    # f2 = get_raga_feature(pitches_unique, pitches_count, breaks_unique, get_dist_btw_notes, c_note, all_notes, asc=True, relax=relax)
    # f2 = get_raga_feature(pitches_unique, pitches_count, breaks_unique, get_dist_btw_notes, c_note, all_notes, asc=False, transpose=True, relax=relax)
    # f3 = get_raga_feature(pitches_unique, pitches_count, breaks_unique, get_strict_dist_btw_notes, c_note, all_notes, asc=True, relax=relax)
    # f4 = get_raga_feature(pitches_unique, pitches_count, breaks_unique, get_strict_dist_btw_notes, c_note, all_notes, asc=False, transpose=True, relax=relax)

    # feat =  np.stack([f1, f2], axis=-1)
    # feat = np.reshape(feat, [12,11,60,2])

    # feat = np.stack([f1, f2, f3, f4], axis=-1)
    # feat = np.reshape(feat, [12,11,60,4])


    return pitch_dist_list_1, pitch_dist_list_2
    # return feat

def get_unique_seq(pitches_arg):
    ps = pitches_arg[0]
    count = 1
    pitches_unique = []
    pitches_count = []
    for p in pitches_arg[1:]:
        if p == ps:
            count += 1
        else:
            pitches_unique.append(ps)
            pitches_count.append(count)
            count = 1
        ps = p

    pitches_unique.append(ps)
    pitches_count.append(count)

    return np.array(pitches_unique), np.array(pitches_count)

def get_raga_feat(pitches):
    raga_feat = np.zeros([12,12,60,4])
    lim = 55
    relax = 2
    pitches_arg = np.argmax(pitches,1)
    pitches_unique, pitches_count = get_unique_seq(pitches_arg)
    pitch_dist_list_1 = get_fast_raga_feature(pitches_unique, pitches_count, asc=True, relax=relax)
    pitch_dist_list_2 = get_fast_raga_feature(pitches_unique, pitches_count, asc=False, relax=relax)

    for i in range(0, 60, 5):
        for j in range(5, lim + 1, 5):
            s = i
            e = (i + j + 60) % 60

            raga_feat[s // 5, e // 5, :, 0] = get_arr(pitch_dist_list_1[(s, e)][0])
            raga_feat[s // 5, e // 5, :, 1] = get_arr(pitch_dist_list_2[(e, s)][0])
            raga_feat[s // 5, e // 5, :, 2] = get_arr(pitch_dist_list_1[(s, e)][1])
            raga_feat[s // 5, e // 5, :, 3] = get_arr(pitch_dist_list_2[(e, s)][1])

    return raga_feat

def gauss_smooth_util(arr):
    smooth = 0
    for i in range(60):
        smooth = smooth + all_notes[i] * arr[i]
    smooth = stadardize(smooth)
    return smooth

def get_arr(arr):
    if len(arr)==0:
        return np.zeros(60)

    return gauss_smooth_util(arr[-1][1:])

def generate_shist_file(pitches):

    pitch_dist_list_1, pitch_dist_list_2 = get_all_raga_features(pitches_arg)

    lim = 55
    raga_feat_dict = {}
    for i in range(0, 60, 5):
        for j in range(0, lim + 1, 5):
            s = i
            e = (i + j + 60) % 60

            if s == e:
                continue

            f.create_dataset('{}_{}_{}_{}'.format(mbid, s, e, 0), data=pitch_dist_list_1[(s, e)][0])
            f.create_dataset('{}_{}_{}_{}'.format(mbid, s, e, 1), data=pitch_dist_list_2[(e, s)][0])
            f.create_dataset('{}_{}_{}_{}'.format(mbid, s, e, 2), data=pitch_dist_list_1[(s, e)][1])
            f.create_dataset('{}_{}_{}_{}'.format(mbid, s, e, 3), data=pitch_dist_list_2[(e, s)][1])


if __name__ == '__main__':
    # pitches_arg = np.arange(0,60)
    # breaks = np.ones(60)
    # get_all_raga_features(pitches_arg, breaks)
    mbid = '99dcaebe-ab49-4fa3-ab6c-a9458143af8e'
    # raga_feat = get_raga_feat_cache(mbid, 0, 1998*5)
    print(raga_feat.shape)
    # range_dict, inv_range_dict = get_range_dict(0,True)
    # print(inv_range_dict)