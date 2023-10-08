import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import trange
import configuration as config


def read_image(file):
    im = Image.open(file).convert("RGB")
    image = np.array(im, dtype=np.uint8)
    return image


def convert_to_gray(image):
    new_img = np.zeros(image.shape[:2], dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            S = np.mean(image[i, j])
            new_img[i, j] = S
    return new_img


def gauss_filter(image, K_size, sigma, full=False):
    pad = K_size // 2
    out = np.zeros((image.shape[0] + pad * 2, image.shape[1] + pad * 2), dtype=float)
    out[pad: pad + image.shape[0], pad: pad + image.shape[1]] = image.copy().astype(float)
    K = np.zeros((K_size, K_size)).astype(float)
    for i in range(-pad, -pad + K_size):
        for j in range(-pad, -pad + K_size):
            K[i, j] = np.exp(-((i - pad) ** 2 + (j - pad) ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    K /= np.sum(np.sqrt(K ** 2))
    if full:
        tmp = out.copy()
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                out[pad + y, pad + x] = int(np.sum(K * tmp[y: y + K_size, x: x + K_size]))
        out = out[pad: pad + image.shape[0], pad: pad + image.shape[1]].astype(np.uint8)
        return out
    return K


def brezenham_circle(image, i, j):
    return np.array(np.concatenate((image[i - 3, j - 1:j + 2], image[i - 2, j + 2],
                                    image[i - 1:i + 2, j + 3], image[i + 2, j + 2], np.flip(image[i + 3, j - 1:j + 2]),
                                    image[i + 2, j - 2], np.flip(image[i - 1:i + 2, j - 3]), image[i - 2, j + 2]),
                                   axis=None))


def comparison(dot, value, more=True):
    if dot < value and more:
        return False
    if dot > value and not more:
        return False
    if dot > value and more:
        return True
    if dot < value and not more:
        return True


def FAST(image, t):
    n = 9
    shift = 3
    H, W = image.shape
    singular_points = np.zeros(shape=(H, W))
    singular_points_coords = []
    for i in range(shift, H - shift):
        for j in range(shift, W - shift):
            t_1 = image[i, j]
            circle = brezenham_circle(image, i, j)
            l = len(circle)
            if (comparison(circle[1], t_1 + t, True) == True and comparison(circle[9], t_1 - t, False) == True) or (
                    comparison(circle[1], t_1 - t, False) == True and comparison(circle[9], t_1 + t, True) == True):
                continue
            if (comparison(circle[5], t_1 + t, True) == True and comparison(circle[13], t_1 - t, False) == True) or (
                    comparison(circle[5], t_1 - t, False) == True and comparison(circle[13], t_1 + t, True) == True):
                continue
            for x in range(-len(circle), len(circle)):
                dots = 0
                more = []
                for y in range(n):
                    if comparison(circle[(x + y) % l], t_1 + t, True):
                        more.append(True)
                        dots += 1
                    elif comparison(circle[(x + y) % l], t_1 - t, False):
                        more.append(False)
                        dots += 1
                    else:
                        break
                if dots == n:
                    ok = True
                    for h in range(n - 1):
                        if (more[h] == True and more[h + 1] == False) or (more[h] == False and more[h + 1] == True):
                            ok = False
                            break
                    if ok:
                        singular_points_coords.append([i, j])
                        singular_points[i, j] = 255
                        break

    return singular_points_coords, singular_points


def Harris(image, singular_points_coords, sigma_for_gauss, k, threshold):
    R_array = []
    K_size_for_gauss = 5
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    for point in singular_points_coords:
        M = np.zeros((2, 2))
        gauss = gauss_filter(image, K_size_for_gauss, sigma_for_gauss)
        for i in range(-2, 3):
            for j in range(-2, 3):
                x, y = point[0] + i, point[1] + j
                G_x, G_y = 0, 0
                out = image[x - 1: x + 2, y - 1: y + 2]
                for h in range(3):
                    for w in range(3):
                        G_y = np.sum(Gy * out)
                        G_x = np.sum(Gx * out)
                A = np.array([[G_x ** 2, G_x * G_y], [G_x * G_y, G_y ** 2]])
                M = M + gauss[i + 2, j + 2] * A
        lamdas = np.linalg.eigvals(M)
        det, trace = lamdas[0] * lamdas[1], lamdas[0] + lamdas[1]
        R = det - k * (trace ** 2)
        if R > 0 and R > threshold:
            R_array.append([point[0], point[1]])
    return R_array


def rotate_mtrx(teta):
    return np.array([[np.cos(teta), np.sin(teta)], [-np.sin(teta), np.cos(teta)]])


def count_centroid_and_angles(image, y, x, patch_size):
    m00, m10, m01 = 0, 0, 0
    for i in range(y - patch_size, y + patch_size + 1):
        for j in range(x - patch_size, x + patch_size - 1):
            if 0 > i or i >= image.shape[0] or 0 > j or j >= image.shape[1]:
                continue
            if (y - i) ** 2 + (x - j) ** 2 <= patch_size ** 2:
                m00 += image[i, j]
                m01 += (y - i) * image[i, j]
                m10 += (x - j) * image[i, j]
    if m00 == 0:
        return np.array([y, x]), 0
    theta = np.arctan2(m01, m10)
    return theta


def get_patterns(n=256, patch_size=31):
    rand1 = np.random.randn(2, n) * patch_size / 5
    rand2 = np.random.randn(2, n) * patch_size / 5
    pattern_points = []
    pattern_points.append(rand1)
    pattern_points.append(rand2)
    return np.array(pattern_points)


def BRIEF(image, patch_size, singular_points_coords, pattern_points):
    image = gauss_filter(image, 5, 1, True)
    descriptors = []
    angles = []
    points = []
    n = 256
    img = image.copy()
    p1, p2 = pattern_points[0], pattern_points[1]
    for k in singular_points_coords:
        if not (patch_size < k[0] < image.shape[0] - patch_size - 1 and patch_size < k[1] < image.shape[
            1] - patch_size - 1):
            continue
        teta_c = count_centroid_and_angles(image, k[0], k[1], patch_size)
        teta_c = np.round(teta_c / (np.pi / 15)) * np.pi / 15
        S1 = (rotate_mtrx(teta_c) @ p1).astype(int)
        S2 = (rotate_mtrx(teta_c) @ p2).astype(int)
        S1[S1 < -patch_size] = -patch_size
        S1[S1 > patch_size] = patch_size
        S2[S2 < -patch_size] = -patch_size
        S2[S2 > patch_size] = patch_size
        bin_row = np.zeros(n, dtype=int)
        for i in range(S1.shape[1]):
            if img[k[0] + S1[0, i], k[1] + S1[1, i]] < img[k[0] + S2[0, i], k[1] + S2[1, i]]:
                bin_row[i] = 1
        descriptors.append(bin_row)
        angles.append(teta_c)
        points.append(k)
    return points, descriptors, angles


# Масштабирование изображения
def resize(image, koef):
    h, w = image.shape
    new_im = np.array(Image.fromarray(image).resize((int(w * koef), int(h * koef))))
    return new_im


# Выводит оcобые точки, углы, дескрипторы
def get_points(image_array, koef, pattern_points):
    image = image_array.copy()
    if koef != 1:
        image = resize(image, koef)
    singular_points_coords, singular_points = FAST(image, 40)
    sorted_singular_points_coords = Harris(image, singular_points_coords, 1, 0.05, 10000000)
    points, descriptors, angles = BRIEF(image, 31, sorted_singular_points_coords, pattern_points)
    return [points, angles, descriptors, koef]


def hamming_distance(coords1, coords2):
    distance = 0
    for i in range(len(coords2)):
        if coords1[i] != coords2[i]:
            distance += 1
    return distance


def Lowe(d_q, d_t, num):
    d_ind = []
    dist = []
    print(f"Полный перебор: {num} из 4")
    time.sleep(0.1)
    for i in trange(len(d_q)):
        temp_arr_full = []
        for j in range(len(d_t)):
            temp_arr_full.append([i, j, hamming_distance(d_q[i], d_t[j])])
            temp_arr_full.sort(key=lambda x: x[2])
        dist.append([temp_arr_full[0], temp_arr_full[1]])
    print(f"Тест Lowe: {num} из 4")
    time.sleep(0.1)
    for i in trange(len(dist)):
        r = dist[i][0][2] / dist[i][1][2]
        if r < 0.8:
            d_ind.append([dist[i][0][0], dist[i][0][1]])
    return d_ind


def get_mtrx(array):
    h1 = get_string(array)
    helper = np.zeros(shape=(len(h1), 6))
    for i in range(0, helper.shape[0], 2):
        helper[i, [0, 1]] = [h1[i], h1[i + 1]]
        helper[i + 1, [2, 3]] = [h1[i], h1[i + 1]]
        helper[i, 4], helper[i + 1, 5] = 1, 1
    return helper


def get_string(array):
    return np.array(array).flatten()


# Берём точки, которые прошли тест Lowe
def get_coords_by_ind(d_ind, q, t):
    q_coords, t_coords = [], []
    d_ind = np.array(d_ind)
    for i in range(len(d_ind)):
        q_coords.append(q[d_ind[i][0]])
        t_coords.append(t[d_ind[i][1]])
    return q_coords, t_coords


def coord_distance(c1, c2):
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def RANSAC(q, t):
    max_inliner = 0
    max_M = []
    res_ind = []
    for i in range(config.N):
        # 3 точки (6 параметров) для оценки параметров аффинного преобразования
        rand = np.random.randint(0, len(q), 3, int)
        # query
        A = get_mtrx([q[rand[0]], q[rand[1]], q[rand[2]]])
        if np.linalg.det(A) == 0:
            continue
        else:
            inliner = 0
            ind = []
            # test
            X = get_string([t[rand[0]], t[rand[1]], t[rand[2]]])
            # СЛАУ
            m = np.linalg.solve(A, X)
            M = m[:4].reshape(2, 2)
            b = m[-2:]
            T = q @ M + b
            for j in range(len(t)):
                d = coord_distance(T[j], t[j])
                if d < config.threshold:
                    inliner += 1
                    ind.append(j)
            if inliner > max_inliner:
                max_inliner = inliner
                max_M = m
                res_ind = ind
    return max_M, res_ind


# Находим x
def get_x(inl_ind, q, t):
    Q, T = [], []
    for i in range(len(inl_ind)):
        Q.append(q[inl_ind[i]])
        T.append(t[inl_ind[i]])
    b = get_string(T)
    A = get_mtrx(Q)
    return np.linalg.pinv(A) @ b


# Оценка параметров искажения
def get_value_distort(q, t, d_ind):
    q_coords, t_coords = get_coords_by_ind(d_ind, q[0], t[0])
    # Масштабирование координат
    for i in range(len(q_coords)):
        q_coords[i] = [q_coords[i][0] * (1 / q[3]), q_coords[i][1] * (1 / q[3])]
        t_coords[i] = [t_coords[i][0] * (1 / t[3]), t_coords[i][1] * (1 / t[3])]
    max_M, inline_ind = RANSAC(q_coords, t_coords)
    X = get_x(inline_ind, q_coords, t_coords)
    return X, inline_ind, max_M


# На одном плоте две картинки
def concatenate_images(image1, image2):
    max_width = 0
    total_height = 0
    for img in [image1, image2]:
        if img.shape[1] > max_width:
            max_width = img.shape[1]
        total_height += img.shape[0]
    final_image = np.zeros((total_height, max_width, 3), dtype=np.uint8)
    current_y = 0
    for image in [image1, image2]:
        image = np.hstack((image, np.zeros((image.shape[0], max_width - image.shape[1], 3))))
        final_image[current_y:current_y + image.shape[0], :, :] = image
        current_y += image.shape[0]
    return final_image


def drawing(image1, query, test, d_ind, M):
    h1, w1, d1 = image1.shape
    Q, T = get_coords_by_ind(d_ind, query[0], test[0])
    draw_polygon('box_in_scene.png', image1, M)
    image2 = read_image('box1.png')
    image3 = concatenate_images(image1, image2)
    for i in range(len(T)):
        T[i] = [T[i][0] + h1, T[i][1]]
    plt.figure()
    for i in range(len(Q)):
        plt.plot([Q[i][1] * (1 / query[3]), T[i][1] * (1 / test[3])],
                 [Q[i][0] * (1 / query[3]), T[i][0] * (1 / test[3])])
    plt.imshow(image3)
    plt.show()


def draw_polygon(file_for_draw, image, max_M):
    im = Image.open(file_for_draw).convert("RGB")
    h, w, d = image.shape
    polygon = [[0, 0], [0, (w - 1)], [h - 1, 0], [h - 1, w - 1]]
    res = []
    max_M = np.array(max_M)
    M = max_M[:4].reshape(2, 2)
    t = max_M[-2:]
    for i in polygon:
        polygon1 = M @ i + t
        res.append(polygon1)
    coords = [(res[2][1], res[2][0]), (res[0][1], res[0][0]),
              (res[1][1], res[1][0]), (res[3][1], res[3][0])]
    img1 = ImageDraw.Draw(im)
    img1.polygon(coords, outline=(255, 0, 0))
    plt.imsave('box1.png', np.array(im))


def apply(img1, img2):
    img1_gray = convert_to_gray(img1)
    img2_gray = convert_to_gray(img2)

    # Случайные 256 точек
    pattern_points = get_patterns(n=256, patch_size=31)

    # Получаем особые точки
    query = get_points(img1_gray, 1, pattern_points)
    test = get_points(img2_gray, 1, pattern_points)
    query_small = get_points(img1_gray, 0.5, pattern_points)
    test_small = get_points(img2_gray, 0.5, pattern_points)
    print("Особые точки получены")

    # Тест Lowe
    match_bb = Lowe(query[2], test[2], 1)
    match_bs = Lowe(query[2], test_small[2], 2)
    match_sb = Lowe(query_small[2], test[2], 3)
    match_ss = Lowe(query_small[2], test_small[2], 4)

    # Ищем лучший мэтч
    lengths = [len(match_bb), len(match_bs), len(match_sb), len(match_ss)]
    max_len = lengths.index(max(lengths))
    print(f'Лучший мэтч: {max_len}')

    # Выбираем лучшие изображения и их мэтч
    query, test, match_ind = [], [], []
    if max_len == 0:
        query = query
        test = test
        match_ind = match_bb
    if max_len == 1:
        query = query
        test = test_small
        match_ind = match_bs
    if max_len == 2:
        query = query_small
        test = test
        match_ind = match_sb
    if max_len == 3:
        query = query_small
        test = test_small
        match_ind = match_ss

    X, inl_ind, M = get_value_distort(query, test, match_ind)
    print(f'Оценка:{X}')
    drawing(img1, query, test, match_ind, M)
