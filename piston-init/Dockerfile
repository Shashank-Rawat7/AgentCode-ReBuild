# Minimal image for the one-shot Piston package-install/verify step.
# curlimages/curl already has curl; jq is added for safe JSON field matching
# instead of ad-hoc text parsing of the packages/runtimes API responses.
FROM curlimages/curl:latest
USER root
RUN apk add --no-cache jq
COPY init.sh /init.sh
RUN chmod +x /init.sh
ENTRYPOINT ["/init.sh"]
