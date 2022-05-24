import json
from webbrowser import get
from matplotlib import pyplot as plt

from study_project.mylib.utils import create_2d_heatmap, get_home_path


PATH = get_home_path()


if __name__ == "__main__":
    # データを読み込み
    with open(PATH + 'data/prop_shap.json', 'r') as f:
        decode_data = json.load(f)

    # ヒートマップを作成
    create_2d_heatmap(decode_data)

    # 追加データをプロット

    # グラフを表示
    plt.grid()
    plt.legend()
    plt.show()
