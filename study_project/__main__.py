from mylib.load_data import LoadData
from mylib.grad_cam import grad_cam
from mylib.model import ModelCreate
from mylib.evaluate_acc import EvaluateAcc


def main():
    acc = EvaluateAcc.model_evaluate(folder_name='org_shap_mis')
    print(acc)


if __name__ == '__main__':
    main()
