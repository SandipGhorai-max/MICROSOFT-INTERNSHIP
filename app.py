File "/usr/local/lib/python3.10/dist-packages/streamlit/runtime/scriptrunner/exec_code.py", line 88, in exec_func_with_error_handling
    result = func()
File "/usr/local/lib/python3.10/dist-packages/streamlit/runtime/scriptrunner/script_runner.py", line 579, in code_to_exec
    exec(code, module._dict_)
File "/content/newApp.py", line 87, in <module>
    main()
File "/content/newApp.py", line 84, in main
    cifar10_classification()
File "/content/newApp.py", line 57, in cifar10_classification
    model = tf.keras.models.load_model('model111.h5')
File "/usr/local/lib/python3.10/dist-packages/keras/src/saving/saving_api.py", line 194, in load_model
    return legacy_h5_format.load_model_from_hdf5(
File "/usr/local/lib/python3.10/dist-packages/keras/src/legacy/saving/legacy_h5_format.py", line 116, in load_model_from_hdf5
    f = h5py.File(filepath, mode="r")
File "/usr/local/lib/python3.10/dist-packages/h5py/hl/files.py", line 561, in __init_
    fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
File "/usr/local/lib/python3.10/dist-packages/h5py/_hl/files.py", line 235, in make_fid
    fid = h5f.open(name, flags, fapl=fapl)
File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
File "h5py/h5f.pyx", line 102, in h5py.h5f.open

mobilenet working but getting this on CIFAR-10