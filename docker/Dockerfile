# Dev image for the box. Explorer builds the same environment as an Apptainer image from the
# same vLLM release (slurm/build_sif.sbatch), so torch / triton / transformers / vLLM match.
FROM vllm/vllm-openai:v0.34.0
ENTRYPOINT []
CMD ["bash"]
COPY docker/requirements.lock docker/requirements-nodeps.txt /tmp/
# Hold every package the image ships at its version; fail instead of silently upgrading torch.
RUN pip freeze | grep -v ' @ ' > /tmp/base-constraints.txt \
 && pip install --no-cache-dir -c /tmp/base-constraints.txt -r /tmp/requirements.lock \
 && pip install --no-cache-dir --no-deps -r /tmp/requirements-nodeps.txt
