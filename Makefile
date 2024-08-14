train:
	python train.py

build:
	cmake -Bbuild -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
	cmake --build build

clean:
	rm -rf build

profile:
	nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop -o profile -f true python train.py

run:
	./build/inference export/model.so

.PHONY: run nsys build clean

