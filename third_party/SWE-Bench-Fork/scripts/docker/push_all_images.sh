#!/bin/bash

DOCKER_NAMESPACE=$1
INSTANCE_ID_FILE=$2

if [ -z "$DOCKER_NAMESPACE" ] || [ -z "$INSTANCE_ID_FILE" ]; then
    echo "Usage: $0 <docker_namespace> <instance_id_file>"
    exit 1
fi

# target namespace
image_list=$(docker image ls --format '{{.Repository}}:{{.Tag}}' | grep sweb | grep -v $DOCKER_NAMESPACE)
instance_ids=$(cat $INSTANCE_ID_FILE)

# KEEP images that are IN the instance_ids
image_list=$(echo "$image_list" | grep -f <(echo "$instance_ids"))

echo "# of images to push: $(echo "$image_list" | wc -l)"

# There are three tiers of images
# - base
# - env
# - eval (instance level)

for image in $image_list; do
    echo "Tagging $image"
    # rename image by replace "__" with "_s_" to comply with docker naming convention
    new_image_name=${image//__/_s_}
    docker tag $image $DOCKER_NAMESPACE/$new_image_name
    echo "Tagged $image to $DOCKER_NAMESPACE/$new_image_name"
    
    docker push $DOCKER_NAMESPACE/$new_image_name
    echo "Pushed $image"
done
