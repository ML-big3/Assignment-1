def linear_regression(X_dataframe, y_dataframe):
	for i in [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000]:
		if X_dataframe.szie>=i:
			X_train, X_test, y_train, y_test = train_test_split(X_dataframe.head(i), y_dataframe.head(i), test_size = 0.2, random_state = 0)
			regressor.fit(X_train, y_train)
			y_pred = regressor.predict(X_test)
    		print("RMSE is" + np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


