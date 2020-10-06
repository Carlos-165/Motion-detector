import cv2
import numpy as numpy

# Captura de video
captura = cv2.VideoCapture('video.mp4')

# Substracción de fondo
sf_mog = cv2.bgsegm.createBackgroundSubtractorMOG(300)
sf_mog2 = cv2.createBackgroundSubtractorMOG2(300, 400, True)
sf_gmg = cv2.bgsegm.createBackgroundSubtractorGMG(10,0.8)
sf_knn = cv2.createBackgroundSubtractorKNN(100,400, True)
sf_cnt = cv2.bgsegm.createBackgroundSubtractorCNT(5, True)

# Número de pixeles para considerar movimiento
movimiento_pix = 2500
mensaje = "Movimiento detectado"

# Conteo de frames
frame_numero = 0

while(True):
    # Regresa valor y el frame actual
    return_valor, frame = captura.read()

    #Chequear si existe el frame actual
    if not return_valor:
        break
    
    frame_numero += 1

    # Redimensionar el frame
    frame_r = cv2.resize(frame,(0,0), fx = 0.3, fy = 0.3)

    # Obtener mask
    mask_mog = sf_mog.apply(frame_r)
    mask_mog2 = sf_mog2.apply(frame_r)
    mask_gmg = sf_gmg.apply(frame_r)
    mask_gmg = cv2.morphologyEx(mask_gmg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    mask_knn = sf_knn.apply(frame_r)
    mask_cnt = sf_cnt.apply(frame_r)

    # Contar todos los pixeles diferentes de 0 en la máscara
    pix_mog = numpy.count_nonzero(mask_mog)
    pix_mog2 = numpy.count_nonzero(mask_mog2)
    pix_gmg = numpy.count_nonzero(mask_gmg)
    pix_knn = numpy.count_nonzero(mask_knn)
    pix_cnt = numpy.count_nonzero(mask_cnt)

    print('Frame mog: %d, Pixel Count: %d' %(frame_numero, pix_mog))
    print('Frame mog2: %d, Pixel Count: %d' %(frame_numero, pix_mog2))
    print('Frame gng: %d, Pixel Count: %d' %(frame_numero, pix_gmg))
    print('Frame knn: %d, Pixel Count: %d' %(frame_numero, pix_knn))
    print('Frame cnt: %d, Pixel Count: %d' %(frame_numero, pix_cnt))

    posicion_titulo = (90,30)

    cv2.putText(mask_mog, 'MOG', posicion_titulo, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(mask_mog, 'MOG2', posicion_titulo, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(mask_mog, 'GNG', posicion_titulo, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(mask_mog, 'KNN', posicion_titulo, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(mask_mog, 'CNT', posicion_titulo, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 2, cv2.LINE_AA)

    if (frame_numero > 1):
        if (pix_mog > movimiento_pix): 
            cv2.putText(mask_mog, 'Movimiento detectado', (5,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        if (pix_mog2 > movimiento_pix): 
            cv2.putText(mask_mog2, 'Movimiento detectado', (5,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        if (pix_gmg > movimiento_pix): 
            cv2.putText(mask_gmg, 'Movimiento detectado', (5,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        if (pix_knn > movimiento_pix): 
            cv2.putText(mask_knn, 'Movimiento detectado', (5,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        if (pix_cnt > movimiento_pix): 
            cv2.putText(mask_cnt, 'Movimiento detectado', (5,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        
    
    cv2.imshow('Original', frame_r)
    cv2.imshow('MOG', mask_mog)
    cv2.imshow('MOG', mask_mog2)
    cv2.imshow('GMG', mask_gmg)
    cv2.imshow('KNN', mask_knn)
    cv2.imshow('CNT', mask_cnt)
    cv2.moveWindow('Original', 0, 0)
    cv2.moveWindow('MOG', 0, 300)
    cv2.moveWindow('KNN', 0, 550)
    cv2.moveWindow('GMG', 650, 0)
    cv2.moveWindow('MOG2', 650, 300)
    cv2.moveWindow('CNT', 650, 550)

    i = cv2.waitKey(10) & 0xff
    if i == 27:
        break

captura.release()
cv2.destroyAllWindows()


