import tensorflow as tf
import pandas as pd

from architecture.models.non_product.Non_product import Non_product as Model
import architecture.models.config as config
import architecture.models.constants as constants


def main():
    tf.set_random_seed(1234)
    all_data = pd.read_csv("./data/unified_processed.csv", index_col="Dataset")
    all_data = all_data.loc[:,"Standardized_Order":"weight"]
    
    # Train set
    for dset in constants.ALLDATA:
        if isinstance(dset, (list,)):
            val_set = dset
            dset = dset[0]
        else:
            val_set = [dset]
        param = config.Config(hidden_size=100,
                              n_layers=2,
                              n_epochs=300,
                              lambd=1e-5,
                              dropout=.1,
                              lr=3e-6,
                              name = dset + "_product") 
        train_indices = [name for name in constants.ALLDATA_SINGLE if 
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
