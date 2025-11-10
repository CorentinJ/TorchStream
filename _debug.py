# def work_fn():
#     a = torch.arange(5)
#     b = torch.cumsum(a, dim=0)
#     c = b // 3
#     d = torch.cumsum(c, dim=0)
#     return d - a


# if __name__ == "__main__":

#     def handler(*args, **kwargs):
#         print("Intercepted call", args, kwargs)
#         return args[0]

#     with SelectivePatch(target="torch.cumsum", dispatch={None: handler}):
#         result = work_fn()
#         print("Result:", result)


from torchstream.sliding_window.kernel_sparsity import get_init_kernel_array

get_init_kernel_array
