import numpy as np

# Métrica MAE - Mean Absolute Error
def mae(y_real, y_predito):
    n = len(y_real)
    absolute = np.abs(y_real - y_predito)
    somatorio = np.sum(absolute)
    return somatorio * (1/n)

# Métrica MSE - Mean Squared Error
def mse(y_real, y_predito):
    n = len(y_real)
    quadrado = (y_real - y_predito)**2
    somatorio = np.sum(quadrado)
    return somatorio * (1/n)

# Métrica RMSE - Root Mean Squared Error
def rmse(y_real, y_predito):
    calculo_mse = mse(y_real, y_predito)
    raiz_mse = np.sqrt(calculo_mse)
    return raiz_mse

# Métrica MSLE - Mean Squared Logarithmic Error
def msle(y_real, y_predito):
    n = len(y_real)
    logaritmo = np.log(y_real + 1) - np.log(y_predito + 1)
    quadrado = logaritmo ** 2
    somatorio = np.sum(quadrado)
    return somatorio * (1/n)

# Métrica RMSLE - Root Squared Logarithmic Error
def rmsle(y_real, y_predito):
    calculo_msle = msle(y_real, y_predito)
    raiz_msle = np.sqrt(calculo_msle)
    return raiz_msle

# 𝑆𝑆𝑟𝑒𝑠 é a soma dos quadrados residuais:
def ss_res(y_real, y_predito):
    quadrado = (y_real - y_predito)**2
    somatorio = np.sum(quadrado)
    return somatorio

# 𝑆𝑆𝑡𝑜𝑡 é a soma total dos quadrados da diferença em relação à média dos valores observados:
def ss_tot(y_real):
    mean_y = np.mean(y_real)
    quadrado = (y_real - mean_y)**2
    somatorio = np.sum(quadrado)
    return somatorio

# Métrica 𝑅2 - Coeficiente de Determinação
def r2(y_real, y_predito):
    calculo_ssres = ss_res(y_real, y_predito)
    calculo_sstot = ss_tot(y_real)
    divisao = calculo_ssres / calculo_sstot
    return 1 - divisao
