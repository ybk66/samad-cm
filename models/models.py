import copy


models = {}


def register(name):#注册模型类
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_spec, args=None, load_sd=False):#创建模型实例
    if args is not None: #如果参数不为空，则更新model_args
        model_args = copy.deepcopy(model_spec['args'])#先拷贝
        model_args.update(args)#再更新

    else:
        model_args = model_spec['args'] #train、test走这里

    
    # model_spec['name']就是sam
    #**model_args将model_args中的参数作为关键字参数传递给模型构造函数
    
    model = models[model_spec['name']](**model_args)
    
    if load_sd:
        model.load_state_dict(model_spec['sd'])
    return model
