import numpy as np
import pandas as pd

def produzir(dataset):
    sub_pasta = f"datasets/UCI_original/UCI HAR Dataset/{dataset}/"
    pasta = sub_pasta+"Inertial Signals/"
    arquivos = (f"body_acc_x_{dataset}.txt", f"body_acc_y_{dataset}.txt", f"body_acc_z_{dataset}.txt", f"body_gyro_x_{dataset}.txt", f"body_gyro_y_{dataset}.txt", f"body_gyro_z_{dataset}.txt", f"total_acc_x_{dataset}.txt", f"total_acc_y_{dataset}.txt", f"total_acc_z_{dataset}.txt")
    vetores = []
    for arquivo in arquivos:
        with open(pasta+arquivo) as f:
            linhas = f.readlines()
            vetores.append(np.array([np.array([float(x) for x in linha.split()]) for linha in linhas]))
    vetores = np.array(vetores)
    # formato do vetor: (9, 7352, 128) -> (7352, 128*9)
    vetores = np.transpose(vetores, (1, 0, 2))

    # concatenar as ultimas dimensoes
    vetores = vetores.reshape(vetores.shape[0], -1)

    with open(sub_pasta+f"y_{dataset}.txt") as f:
        linhas = f.readlines()
        y = np.array([int(x)-1 for x in linhas])

    # concatena y ao final
    y = y.reshape(-1, 1)
    vetores = np.concatenate((vetores, y), axis=1)
    
    # slvar vetores como train.csv
    df = pd.DataFrame(vetores)
    df.to_csv(f"datasets/UCI_original/{dataset}.csv", index=False, header=False)

produzir("train")
produzir("test")