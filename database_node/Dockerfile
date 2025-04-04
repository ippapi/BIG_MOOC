# Start with a base image
FROM openjdk:11-jre-slim

# Install necessary dependencies (you can add others as needed)
RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    git \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Install Kafka (example)
RUN curl -s https://archive.apache.org/dist/kafka/2.7.0/kafka_2.13-2.7.0.tgz | tar -xvzf - -C /opt \
    && ln -s /opt/kafka_2.13-2.7.0 /opt/kafka

# Install Cassandra (example)
RUN curl -sSL https://downloads.apache.org/cassandra/debian/39x/cassandra-4.0.0-bin.tar.gz | tar -xvzf - -C /opt \
    && ln -s /opt/cassandra-4.0.0 /opt/cassandra

# Set environment variables for Kafka and Cassandra
ENV KAFKA_HOME=/opt/kafka
ENV CASSANDRA_HOME=/opt/cassandra

# Copy your scripts (like data loader) into the image
COPY ./dataloader /usr/local/bin/dataloader

# Expose ports for your services (Kafka, Cassandra)
EXPOSE 9092 2181 7000 9042

# Start the services (Cassandra, Kafka, and the dataloader script)
CMD ["bash", "-c", "service cassandra start && /opt/kafka/bin/kafka-server-start.sh /opt/kafka/config/server.properties & /usr/local/bin/dataloader"]
