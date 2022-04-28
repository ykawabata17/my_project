from mylib.load_data import LoadData
from mylib.grad_cam import grad_cam
from mylib.model import ModelCreate


def main():
    model_create = ModelCreate()
    for i in range(11):
        model_create.model_org_shap_mis(org_num=10000, shap_mis_num=i*1000, file_name=i*1000)


if __name__ == '__main__':
    main()
