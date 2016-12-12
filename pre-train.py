import scipy.io
pre_train_data = scipy.io.loadmat(r'C:\Users\liuxi\Documents\science paper codes\before-pre-train.mat')
for every_entry in pre_train_data.keys():
    exec(every_entry + " = pre_train_data[" +" '" +every_entry + "' " + "]")

input("Press Enter to continue...")
