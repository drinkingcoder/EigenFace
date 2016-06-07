mytest mytrain:
	g++ `pkg-config --cflags opencv` -o mytest mytest.cpp `pkg-config --libs opencv`
	g++ `pkg-config --cflags opencv` -o mytrain mytrain.cpp `pkg-config --libs opencv`
clean:
	rm mytest
	rm mytrain
