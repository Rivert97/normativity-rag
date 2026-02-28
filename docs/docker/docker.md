# Docker

Follow this steps if you want to use Docker:

1. Install Docker.
2. Run the following command to pull the image:

    ```bash
    docker pull Rivert97/normativity-rag:latest
    ```

    > __NOTE:__ You can also build your own image using the Dockerfile in the root directory. First create a llama_cpp image with the `Dockerfile.llama` and then the normativity-rag image with the `Dockerfile`.

3. Run the container to extract the embeddings of the documents:

    ```bash
    docker run -v /home/$USER/documents:/documents -v /home/$USER/db:/db -env-file .env Rivert97/normativity-rag:latest extract -c CUSTOM_COLLECTION -d /documents --database-dir /db
    ```

    > __NOTE:__ The documents should be placed in the host machine in /home/$USER/documents and the database will be stored in /home/$USER/db.

4. Run the container to chat with the documents:

    ```bash
    docker run -v /home/$USER/documents:/documents -v /home/$USER/db:/db -env-file .env Rivert97/normativity-rag:latest chat -c CUSTOM_COLLECTION --database-dir /db --show-context
    ```

> __IMPORTANT:__ The docker image has an entrypoint equivalent to `python run.py`. This means that all the functionalities described for the scripts work for the containers by simply passing the correct options (e.g. `docker run ... extract -h` to see the help message).
