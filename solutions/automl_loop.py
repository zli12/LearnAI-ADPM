all_output = {'y_1': {'local_run': local_run, 'model': model, 'cl_report': cl_report}}

for i in range(1, 4): ## loop over each target class
    print("Running automl on y_" + str(i+1))
    automl_config.fit_params.update({'y': y_train.values[:, i]})
    local_run = experiment.submit(automl_config, show_output=True)
    best_run, fitted_model = local_run.get_output()
    y_pred = fitted_model.predict(X_test)
    cl_report = classification_report(y_test.values[:, i], y_pred)
    print(cl_report)
    all_output['y_' + str(i+1)] = {'local_run': local_run, 'cl_report': cl_report}
    description = 'autometed ML PdM (predict y_{})'.format(str(i+1))
    model = local_run.register_model(description=description, tags=None)