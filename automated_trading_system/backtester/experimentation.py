class A:
    def __init__(self, arg_to_class, shared_data):
        self.data = arg_to_class
        self.shared_data = shared_data


class Container:
    def __init__(self):
        self.shared_data = {}

    def set_child(self, child_cls, **kwargs):
        self.child = child_cls(kwargs, self.shared_data)


container = Container()

container.set_child(A, arg_to_class="ARGUMENT")

container.shared_data["test"] = 42

print(container.child.data)
print(container.child.shared_data["test"])

if container.shared_data == container.child.shared_data:
    print("objects are the same")
else:
    print("objects are NOT the same")
