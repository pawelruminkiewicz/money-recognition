import np as np
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.cbook import flatten
from numpy.core.tests.test_mem_overlap import xrange
from copy import deepcopy

#pomocnicze wyswietlenie histogramu
def plot_hist(img):
    histo, x = np.histogram(img, range(0, 256), density=True)
    plt.plot(histo)
    plt.show()

#funkcja liczaca przesuniecie punktu po wektorach
def wektorkiKarola(mniejszeKontury,wspolczynnikOdSrodka):
    for f in range(len(contourList)):
        for _f in range(len(contourList[f])):
            xPunktu = contourList[f][_f][0][0]
            yPunktu = contourList[f][_f][0][1]
            xSrodka = prawilnyXSrodka[f]
            ySrodka = prawilnyYSrodka[f]
            mniejszeKontury[f][_f][0][0] = xSrodka + ((xSrodka - xPunktu) * wspolczynnikOdSrodka*(-1))
            mniejszeKontury[f][_f][0][1] = ySrodka + ((ySrodka - yPunktu) * wspolczynnikOdSrodka*(-1))
    return mniejszeKontury

if __name__ == '__main__':
    zmienna = "010_pop"
    src = cv2.imread("wzorce/"+zmienna+".jpg", 1)
    #plot_hist(src)
    src2 = src.copy()
    cvv = src.copy()
    cvv2 = cv2.imread("wzorce/"+zmienna+".jpg", -1)

    #Sprawdzenie tla - ciemne czy biale
    hist = cv2.calcHist([src], [0], None, [256], [0, 256])
    hist = [val[0] for val in hist];
    indices = list(range(0, 256));
    s = [(x, y) for y, x in sorted(zip(hist, indices), reverse=True)]
    index_of_highest_peak = s[0][0];
    index_of_second_highest_peak = s[1][0];
    #wspolczynnik bieli - sprawdzamy gdzie jest wierzcholek hista
    wspolczynnikBieli = (index_of_highest_peak+index_of_second_highest_peak)/2
    print(wspolczynnikBieli)
    #Dodajemy szum do oryginalnego zdjecia
    img = cv2.blur(src, (4, 4))
    img = cv2.GaussianBlur(img, (35, 35), 0)
    #img = cv2.medianBlur(img, 25)
    #img = cv2.equalizeHist(img)
    #img = cv2.Laplacian(img, cv2.CV_8UC1, ksize=5)
    #img = cv2.Sobel(img, cv2.CV_8U, 1, 1, ksize=5)
    #bk, img = cv2.threshold(img, 20, 80, cv2.THRESH_BINARY)

    #konwertujemy fotke na odcienie szarosci
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("toGrey.png",img)
    #sprawdzamy wspolczynnik bieli i dostosowujemy thresha plus dla bieli negatyw
    if(wspolczynnikBieli>75):
        ret, thresh = cv2.threshold(img, 110, 255, 0)
        thresh=255-thresh
    else:
        ret, thresh = cv2.threshold(img, 55, 255, 0)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contourList = []
    tempCont = []
    tempRadius = []
    tempX = []
    tempY = []
    tempMaxRadius = []
    sredniDist = []

    #sprawdzamy czy namierzone obiekty to kola
    #liczymy momenty aby uzyskac srodek ciezkosci
    for i in range(len(contours)):
        if(hierarchy[0, i, 3]==-1): #to zalatwia kontury wewnatrz konturow
            M = cv2.moments(contours[i])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                continue #bo nie mozemy dzielic przez zero
            #uzupelniamy tablice odleglosci miedzy wierzcholkami konturu a srodkiem
            for u in contours[i]:
                aX = u[0][0]
                aY = u[0][1]
                dist = np.sqrt((cX - aX) ** 2 + (cY - aY) ** 2)
                sredniDist.append(dist)
            #liczymy srednia odleglosc
            sredniaZSredniegoDista = np.mean(sredniDist)
            it = 0

            #sprawdzamy odleglosc wierzcholkow wzgledem sredniej odleglosci
            for m in contours[i]:
                aX = m[0][0]
                aY = m[0][1]
                dist = np.sqrt((cX - aX) ** 2 + (cY - aY) ** 2)
                if(dist>sredniaZSredniegoDista):
                    pom=sredniaZSredniegoDista/dist
                else:
                    pom = dist / sredniaZSredniegoDista
                if(pom>0.8):
                    it = it+1

            if (it > len(contours[i])):
                pom2 = len(contours[i]) / it
            else:
                pom2 = it / len(contours[i])
            #sprawdzamy czy min 90proc wierzcholkow spelnia powyzsze warunki
            if(pom2>0.9):
                tempCont.append(contours[i])
                tempRadius.append(sredniaZSredniegoDista)
                tempX.append(cX)
                tempY.append(cY)
                tempMaxRadius.append(max(sredniDist))
            sredniDist.clear()

    prawilnyXSrodka = []
    prawilnyYSrodka = []
    prawilnyPromien = []
    for o in range(len(tempRadius)):
        if(tempRadius[o]/max(tempRadius)>0.4):
            contourList.append(tempCont[o])
            prawilnyXSrodka.append(tempX[o])
            prawilnyYSrodka.append(tempY[o])
            prawilnyPromien.append(tempRadius[o])
            #cv2.circle(cvv, (tempX[o], tempY[o]), 2, (255, 255, 0), 3, cv2.LINE_AA)
            #cv2.circle(src2, (tempX[o], tempY[o]), int(tempMaxRadius[o]), (105, 255, 0), 3, cv2.LINE_AA)
            #cv2.circle(src2, (tempX[o], tempY[o]), int(tempMaxRadius[o] * 0.7), (105, 255, 0), 3, cv2.LINE_AA)
            #cv2.circle(src2, (tempX[o], tempY[o]), int(tempMaxRadius[o] * 0.5), (105, 255, 0), 3, cv2.LINE_AA)


    mniejszeKontury_0_9 = deepcopy(contourList)
    mniejszeKontury_0_5 = deepcopy(contourList)
    mniejszeKontury_0_9 = wektorkiKarola(mniejszeKontury_0_9, 0.7)
    mniejszeKontury_0_5 = wektorkiKarola(mniejszeKontury_0_5, 0.5)

    # maski zer do wycinania

    #do usuniecia newContourList = np.array(contourList)
    channel_count = cvv2.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count


    for obiekt in range(len(contourList)):
        lewy=obiekt
        prawy=obiekt+1
        mask1 = np.zeros(cvv2.shape, dtype=np.uint8)
        mask2 = np.zeros(cvv2.shape, dtype=np.uint8)
        mask3 = np.zeros(cvv2.shape, dtype=np.uint8)

        cv2.fillPoly(mask1, contourList[lewy:prawy], ignore_mask_color)
        masked_image2 = cv2.bitwise_and(cvv2, mask1)

        cv2.fillPoly(mask2, mniejszeKontury_0_9[lewy:prawy], ignore_mask_color)
        masked_image1 = cv2.bitwise_and(cvv2, mask2)

        masked_image = masked_image2-masked_image1


        cv2.fillPoly(mask3, mniejszeKontury_0_5[lewy:prawy], ignore_mask_color)
        masked_image_in=cv2.bitwise_and(cvv2, mask3)

        masked_hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
        masked_hsv_in = cv2.cvtColor(masked_image_in, cv2.COLOR_BGR2HSV)

        zew1 = round(np.mean(masked_hsv[masked_hsv[:, :, 2] > 0, 0]))
        wew1 = round(np.mean(masked_hsv_in[masked_hsv_in[:, :, 2] > 0, 0]))
        zew2 = round(np.mean(masked_hsv[masked_hsv[:, :, 2] > 0, 1]))
        wew2 = round(np.mean(masked_hsv_in[masked_hsv_in[:, :, 2] > 0, 1]))
        zew3 = round(np.mean(masked_hsv[masked_hsv[:, :, 2] > 0, 2]))
        wew3 = round(np.mean(masked_hsv_in[masked_hsv_in[:, :, 2] > 0, 2]))

        #2zl lub 5zl
        if(abs(zew2-wew2)>=37):
            if(wew2>zew2):
                cv2.putText(src,"5zl",(prawilnyXSrodka[obiekt], prawilnyYSrodka[obiekt]), cv2.FONT_HERSHEY_SIMPLEX, 3,(0, 0, 255), 10)
            else:
                cv2.putText(src, "2zl", (prawilnyXSrodka[obiekt], prawilnyYSrodka[obiekt]),cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10)
        else:
            if(prawilnyPromien[obiekt]>max(prawilnyPromien)*0.92) and (zew2<90 or wew2<90):
                cv2.putText(src, "1zl", (prawilnyXSrodka[obiekt], prawilnyYSrodka[obiekt] ),cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10)


            elif (prawilnyPromien[obiekt] < min(prawilnyPromien) * 1.1) and  (zew2 < 90 or wew2 < 90):
                cv2.putText(src, "10gr", (prawilnyXSrodka[obiekt], prawilnyYSrodka[obiekt]),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10)
            elif (zew2 > 70 and wew2 > 70):
                cv2.putText(src, "5gr", (prawilnyXSrodka[obiekt], prawilnyYSrodka[obiekt]),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10)
            else:
                cv2.putText(src, " ", (prawilnyXSrodka[obiekt], prawilnyYSrodka[obiekt]),cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10)

        '''

        zew1 = round(np.mean(masked_image[masked_hsv[:, :, 2] > 0, 0]))
        wew1 = round(np.mean(masked_image_in[masked_hsv_in[:, :, 2] > 0, 0]))
        zew2 = round(np.mean(masked_image[masked_hsv[:, :, 2] > 0, 1]))
        wew2 = round(np.mean(masked_image_in[masked_hsv_in[:, :, 2] > 0, 1]))
        zew3 = round(np.mean(masked_image[masked_hsv[:, :, 2] > 0, 2]))
        wew3 = round(np.mean(masked_image_in[masked_hsv_in[:, :, 2] > 0, 2]))
        '''
        #cv2.imwrite('ou_' + str(lewy) + "-" + str(prawy) + ".png", masked_image)
        #cv2.imwrite('in_'+str(lewy)+"-"+str(prawy)+".png", masked_image_in)

        #cv2.putText(src, "["+str(wew1)+","+str(zew1)+"]"+"["+str(wew2)+","+str(zew2)+"]"+"["+str(wew3)+","+str(zew3)+"]", (prawilnyXSrodka[obiekt] - 20, prawilnyYSrodka[obiekt] - 90),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 2)




    #masked_hsv[masked_hsv[:,:,2]>0,1] = 50
   # masked_hsv[masked_hsv[:,:,2]>0, 2] = 50
    #masked_image = cv2.cvtColor(masked_image, cv2.COLOR_HSV2BGR)
    '''srednia = []
    for row in masked_hsv:
        for (h, s, v) in row:
            # Only our pixels, not added black background
            if (h != 0 and s != 0 and v != 0):
                srednia.append(h)
    print(np.median(srednia))'''

    # save the result
    #cv2.imwrite('image_masked2.png', masked_image)
   # cv2.imwrite('image_masked2_in.png', masked_image_in)



    cv2.drawContours(src, contourList, -1, (0, 0, 255), 7)

    #cv2.imwrite("wektorki.png",cvv)
    #cv2.imwrite("new_mon.png",src2)
    #cv2.imwrite("new_mon_filtr1.png",thresh)
    cv2.imwrite("result/result_"+zmienna+".png", src)