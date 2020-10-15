# BODA

Source code for the core BODA library is seperated into 4 submodules:

1. boda.data - Specifies the data upon which a model is built
2. boda.model - Specifies the predictive model associated with the underlying data
3. boda.graph - Specifies computation graphs on top of predictive models (e.g., for training and inference)
4. boda.generator - Specifies methods for generating sequences

# Philosophy

Modeling projects are hierarchical. Your data determines which types of models you can use, and your model determines what computations you can do. In that spirit we assemble a `System` utilizing Python inheritance. Basically:

```
def get_assembler(Data, Model, Graph):
    class Assembler(Graph, Model, Data):
        def __init__(self,**kwargs):
            super().__init__(**kwargs)
            self.save_hyperparameters()
    return Assembler

my_assembler = get_assembler(Data, Model, Graph)
my_system    = my_assember(**kwargs)
```
 Where `Data`, `Model`, and `Graph` are classes we've specified in the corresponding submodule.
