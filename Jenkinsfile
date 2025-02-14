pipeline {
    agent any

    environment {
        CLEAN_IMAGE = 'clean-data:latest'
    }

    stages {
        stage('Build Docker Clean-Data') {
            steps {
                script {
                    bat 'docker build -t %CLEAN_IMAGE% .'
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
                archiveArtifacts artifacts: 'work/models/*.pkl', fingerprint: true
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
