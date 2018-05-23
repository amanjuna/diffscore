import tensorflow as tf
import pandas as pd

from architecture.models.product.Product import Product as Model
import architecture.models.config as config

def main():
    tf.set_random_seed(1234)
    all_data = pd.read_csv("./data/unified_processed.csv", index_col="Dataset")
    all_data = all_data.loc[:,"Standardized_Order":"weight"]
    
    # Train set
    for dset in config.ALLDATA:
        if isinstance(dset, (list,)):
            val_set = dset
            dset = dset[0]
        else:
            val_set = [dset]
        param = config.Config(hidden_size=300,
                              n_layers=3, 
                              n_epochs=200,  
                              beta=1, 
                              lambd=1, 
                              lr=3e-5,
                              name = dset + "_combined") 
        train_indices = [name for name in config.ALLDATA_SINGLE if 
                             name not in val_set]
        train_data = all_data.loc[train_indices, :]
        val_data = all_data.loc[val_set, :]
        if len(val_data) == 0:
            continue
        # Fit and log model
        model = Model(param, None)
        model.fit(train_data, val_data)
        print(model.predict(val_data))
        tf.reset_default_graph()

        
if __name__ == "__main__":
    main()
