## 이미지 input
train.py 
eval.py 
Average PCE: 0.6778571428571428
-> Average PCE: 0.695


## 2D pose - 1
normalize_screen_coordinates
2D pose는 screen 좌표계 정규화를 한다.
2D pose만 입력으로 받는다.
Dataset Split 1 => Average PCE: 0.7435714285714285
Dataset Split 2 => Average PCE: 0.7317857142857143
Dataset Split 3 => Average PCE: 0.725
Dataset Split 4 => Average PCE: 0.7317857142857143
Total Average PCEs: 0.7330

## 2D pose - 2
normalize_screen_coordinates
2D pose는 screen 좌표계 정규화를 한다.
2D pose+conf를 입력으로 받는다.
Dataset Split 1 => Average PCE: 0.7392857142857143
Dataset Split 2 => Average PCE: 0.7282142857142857
Dataset Split 3 => Average PCE: 0.7225
Dataset Split 4 => Average PCE: 0.7296428571428571
Total Average PCEs: 0.7299

## 2D pose - 3
normalize_CanonPose
CanonPose에서 소개된 2D pose 정규화를 수행한다 
2D pose만 입력으로 받는다.
Dataset Split 1 => Average PCE: 0.7292857142857143
Dataset Split 2 => Average PCE: 0.7153571428571428
Dataset Split 3 => Average PCE: 0.7053571428571429
Dataset Split 4 => Average PCE: 0.7128571428571429
Total Average PCEs: 0.7157

## 2D pose - 4
normalize_CanonPose
CanonPose에서 소개된 2D pose 정규화를 수행한다 
2D pose+conf를 입력으로 받는다.
Dataset Split 1 => Average PCE: 0.7257142857142858
Dataset Split 2 => Average PCE: 0.7178571428571429
Dataset Split 3 => Average PCE: 0.7067857142857142
Dataset Split 4 => Average PCE: 0.7125
Total Average PCEs: 0.7157

## 2D pose - 5
mode 1 + liftingNet
Dataset Split 1 => Average PCE: 0.7475
Dataset Split 2 => Average PCE: 0.8214285714285714
Dataset Split 3 => Average PCE: 0.81
Dataset Split 4 => Average PCE: 0.8014285714285714
Total Average PCEs: 0.7951


## 2D pose - 6
mode 2 + liftingNet
Dataset Split 1 => Average PCE: 0.7496428571428572
Dataset Split 2 => Average PCE: 0.8285714285714286
Dataset Split 3 => Average PCE: 0.8121428571428572
Dataset Split 4 => Average PCE: 0.8060714285714285

## 2D pose - 7
mode 3 + liftingNet
Dataset Split 1 => Average PCE: 0.7632142857142857
Dataset Split 2 => Average PCE: 0.8032142857142858
Dataset Split 3 => Average PCE: 0.79
Dataset Split 4 => Average PCE: 0.7932142857142858

## 2D pose - 8
mode 4 + liftingNet
Dataset Split 1 => Average PCE: 0.7478571428571429
Dataset Split 2 => Average PCE: 0.7982142857142858
Dataset Split 3 => Average PCE: 0.7939285714285714
Dataset Split 4 => Average PCE: 0.7878571428571428


## 3D pose 
PoseFormerV2 

Dataset Split 1 => Average PCE: 0.7239285714285715
Dataset Split 2 => Average PCE: 0.7260714285714286
Dataset Split 3 => Average PCE: 0.7275
Dataset Split 4 => Average PCE: 0.7292857142857143
Total Average PCEs: 0.7267