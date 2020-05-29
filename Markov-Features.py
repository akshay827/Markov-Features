def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def round_and_abs(a):
    return np.absolute(np.round(a))

#Note: Try mode="db4" incase image size is small
def meyer_dwt(im, mode="dmey"):
    coeffs3 = pywt.wavedec2(im, mode, level=3)
    coeffs2 = pywt.wavedec2(im, mode, level=2)
    coeffs1 = pywt.wavedec2(im, mode, level=1)
   
    # [A3, H3, V3, D3, A2, H2, V2, D2, A1, H1, V1, D1]
    coeffs_list = []
    for coeffs in [coeffs3, coeffs2, coeffs1]:
        coeffs_list.append(coeffs[0])
        for i in range(3):
            coeffs_list.append(coeffs[1][i])
    
    #round and abs the coeffs
    for i in range(len(coeffs_list)):
        coeffs_list[i] = round_and_abs(coeffs_list[i])
        
    return coeffs_list

def H_diff_array (a, diff):
    a_up = np.roll(a, diff, axis=0)
    ah = np.subtract(a, a_up)[:diff,:]
    return ah
    
def V_diff_array (a, diff):
    a_left = np.roll(a, diff, axis=1)
    av = np.subtract(a, a_left)[:,:diff]
    return av

def threshold_array(a, T):
    a[a>T] = T
    a[a<-T] = -T
    return a

def H_markov(a, T):
    c = np.arange(-T,T+1)
    m = Markov(a, classes=c)
    return m.p

def V_markov(a, T):
    c = np.arange(-T,T+1)
    m = Markov(np.transpose(a), classes=c)
    return m.p

#Dependency across positions
def extract_pos_features(coeffs_list):
    pos_dwt_features = []
    for coeff in coeffs_list:

        coeff_h = H_diff_array(coeff, -1)
        coeff_v = V_diff_array(coeff, -1)

        coeff_h = threshold_array(coeff_h, 4)
        coeff_v = threshold_array(coeff_v, 4)

        #transition probability matrices:
        coeff_P1h = H_markov(coeff_h, 4).flatten()
        coeff_P1v = V_markov(coeff_h, 4).flatten()
        coeff_P2h = H_markov(coeff_v, 4).flatten()
        coeff_P2v = V_markov(coeff_v, 4).flatten()

        coeff_features = np.concatenate( (coeff_P1h, coeff_P1v, coeff_P2h, coeff_P2v), axis=0) 

        pos_dwt_features.append(coeff_features)
    
    pos_dwt_features = np.array(pos_dwt_features)

def difference_like_array(coeffs_list):
    diff_like_array = []
    for i in range(9, 12):

        C1 = coeffs_list[i]
        C2 = coeffs_list[i - 4]
        C3 = coeffs_list[i - 8]
        
        
        '''
        Getting index out of bound error otherwise
        '''
        C2_new = np.zeros((int(C1.shape[0] / 2), int(C1.shape[1] / 2)))
        C3_new = np.zeros((int(C2.shape[0] / 2), int(C2.shape[1] / 2)))

        for x in range(C2_new.shape[0]):
            for y in range(C2_new.shape[1]):

                C2_new[x][y] = np.round((C1[2*x - 1][2*y - 1] + C1[2*x - 1][2*y] + 
                                            C1[2*x][2*y - 1] + C1[2*x][2*y]) / 4) - C2[x][y]
        for x in range(C3_new.shape[0]):
            for y in range(C3_new.shape[1]):
                C3_new[x][y] = np.round((C2[2*x - 1][2*y - 1] + C2[2*x - 1][2*y] + 
                                            C2[2*x][2*y - 1] + C2[2*x][2*y]) / 4) - C3[x][y]
        diff_like_array.append(C2_new)
        diff_like_array.append(C3_new)
        
    return np.array(diff_like_array)
    return (np.concatenate(pos_dwt_features, axis = 0))

def extract_scale_features(coeffs_list):
    
    diff_like_array = difference_like_array(coeffs_list)
    scale_dwt_features = []
    for arr in diff_like_array:
        arr = threshold_array(arr, 4)

        #transition probability matrices:
        arr_P1h = H_markov(arr, 4).flatten()
        arr_P1v = V_markov(arr, 4).flatten()

        arr_features = np.concatenate( (arr_P1h, arr_P1v), axis=0)

        scale_dwt_features.append(arr_features)
        
    scale_dwt_features = np.array(scale_dwt_features)    
    return np.concatenate(scale_dwt_features, axis = 0)

#Dependency across orientations

def cross_diff_array(coeffs_list):
    cross_differences = []
    for i in range(1, 10, 4):
        HV_diff = coeffs_list[i] - coeffs_list[i + 1]
        VD_diff = coeffs_list[i + 1] - coeffs_list[i + 2]
        DH_diff = coeffs_list[i + 2] - coeffs_list[i]

        cross_differences.append(HV_diff)
        cross_differences.append(VD_diff)
        cross_differences.append(DH_diff)

    return np.array(cross_differences)

def extract_orient_features(coeffs_list):
    #[HV1, VD1, DH1, HV2, VD2, DH2, HV3, VD3, DH3]
    cross_differences = cross_diff_array(coeffs_list)

    orient_dwt_features = []
    for diff in cross_differences:

        diff = threshold_array(diff, 4)

        #transition probability matrices:
        diff_P1h = H_markov(diff, 4).flatten()
        diff_P1v = V_markov(diff, 4).flatten()

        diff_features = np.concatenate( (diff_P1h, diff_P1v), axis=0)

        orient_dwt_features.append(diff_features)
        
    orient_dwt_features = np.array(orient_dwt_features)
    return np.concatenate(orient_dwt_features, axis = 0)

def extract_DWT_features(x):
    dwt_features = []
    for i in range (0, x.shape[0]):
        im = x[i,:,:]
        
        # [A3, H3, V3, D3, A2, H2, V2, D2, A1, H1, V1, D1]
        coeffs_list = meyer_dwt(im)
        coeffs_list.append(im) 
       
        #Dependency across positions
        pos_dwt_features = extract_pos_features(coeffs_list)
        
        #Dependency across scales
        scale_dwt_features = extract_scale_features(coeffs_list)
        
        #Dependency across orientation
        orient_dwt_features = extract_orient_features(coeffs_list)
        
        features = np.concatenate( (pos_dwt_features, scale_dwt_features, orient_dwt_features), axis=0)
        dwt_features.append(features)
    
    return np.array(dwt_features)

def extract_DCT_features(x):
    dct_features = []
    for i in range (0, x.shape[0]):
        
        im = x[i,:,:]
        imsize = im.shape
        dct = np.zeros(imsize)

        # Do 8x8 DCT on image (in-place)
        for i in r_[:imsize[0]:8]:
            for j in r_[:imsize[1]:8]:
                dct[i:(i+8),j:(j+8)] = dct2( im[i:(i+8),j:(j+8)] )
        dct = round_and_abs (dct)
        
        
        F = dct
        Fh = H_diff_array(F, -1)
        Fv = V_diff_array(F, -1)
        Fh = threshold_array(Fh, 4)
        Fv = threshold_array(Fv, 4)
        P1h = H_markov(Fh, 4).flatten()
        P1v = V_markov(Fh, 4).flatten()
        P2h = H_markov(Fv, 4).flatten()
        P2v = V_markov(Fv, 4).flatten()
        
        features = np.concatenate( (P1h, P1v, P2h, P2v), axis=0) 
        dct_features.append(features)
    return np.array(dct_features)

#x is a numpy array of images of shape (num_images, im_height, im_width, im_channels)
def extract_features(x):
    return extract_DWT_features(x), extract_DCT_features(x)
