docker build -t nnopt/base .
echo "Finished building image"
docker run -d -p 5000:5000 registry:2
docker tag nnopt/base localhost:5000/nnopt/base
docker push localhost:5000/nnopt/base
SINGULARITY_NOHTTPS=1 singularity build nnopt.simg docker://localhost:5000/nnopt/base
