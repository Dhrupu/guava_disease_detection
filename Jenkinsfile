pipeline {
    agent any

    environment {
        IMAGE_NAME = "guava-app"
        CONTAINER_NAME = "guava-container"
    }

    stages {

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t $IMAGE_NAME .'
            }
        }

        stage('Stop Old Container') {
            steps {
                sh '''
                docker stop $CONTAINER_NAME || true
                docker rm $CONTAINER_NAME || true
                '''
            }
        }

        stage('Run Container') {
            steps {
                sh '''
                docker run -d \
                --name $CONTAINER_NAME \
                -p 80:5000 \
                $IMAGE_NAME
                '''
            }
        }

    }
}