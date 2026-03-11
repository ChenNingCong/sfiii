# Use the official diambra engine as the base
FROM diambra/engine:latest

# Define the absolute target path
ENV TARGET_PATH=/bin/diambraEngineServer

# 1. Remove the existing file to ensure a clean slate
# 2. Copy the new binary from your local context
# 3. Set permissions
RUN rm -f ${TARGET_PATH}
COPY ./binary/diambraEngineServer ${TARGET_PATH}

# Explicitly set the entrypoint
ENTRYPOINT ["/bin/diambraEngineServer"]