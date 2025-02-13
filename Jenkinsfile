pipeline {
    agent any

    environment {
        CLEAN_IMAGE = 'clean-data:latest'
    }

    stages {
        stage('Cloner le repo') {
            steps {
                bat 'git clone https://github.com/allanTous23/kubeflow.git'
            }
        }
        
        stage('Build Docker Clean-Data') {
            steps {
                script {
                    sh 'docker build -t $CLEAN_IMAGE .'
                }
            }
        }

        stage('Nettoyer les données') {
            steps {
                script {
                    bat 'docker run --rm clean-data:latest'
                }
            }
        }

        stage('Sauvegarder les artefacts') {
            steps {
                archiveArtifacts artifacts: 'models/*.pkl', fingerprint: true
            }
        }
    }

    post {
        success {
            echo "Pipeline terminé avec succès"
        }
        failure {
            echo "Échec du pipeline"
        }
    }
}
