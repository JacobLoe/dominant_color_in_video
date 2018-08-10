build:
	sudo docker build -t chrstn_hntschl/dc_extractor:0.1 -f Dockerfile ../

shell: build
	sudo docker run --rm -it -v /home/datasets/AdA/:/data chrstn_hntschl/dc_extractor:0.1 /bin/bash
