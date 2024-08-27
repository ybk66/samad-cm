import copy


datasets = {}


def register(name):#注册数据集类
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(dataset_spec, args=None):#创建数据集实例
    if args is not None:

        dataset_args = copy.deepcopy(dataset_spec['args'])
        dataset_args.update(args)
    else:

        dataset_args = dataset_spec['args']
        
    dataset = datasets[dataset_spec['name']](**dataset_args)
    return dataset
