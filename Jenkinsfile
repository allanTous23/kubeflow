pipeline {
    agent any

    environment {
        CLEAN_IMAGE = 'mlops:latest'
    }

    stages {
        
        stage('creation repertoire travail') {
            steps {
             bat 'if not exist model_artifacts mkdir model_artifacts'
            }
        }
        
        stage('Build Docker Clean-Data qsdfqdsfqsdfsdq') {
            steps {
                script {
                    bat 'docker build -t %CLEAN_IMAGE% .'
                }
            }
        }

        stage('Nettoyer les données sdfsdfsdfsdfsdfsdf  dd') {
            steps {
                script {
                    bat 'docker run --rm %CLEAN_IMAGE% '
                }
            }
        }

        stage('Démarrer les services') {
            steps {
                bat 'docker-compose up -d'
            }
        }

        stage('Sauvegarder les artefacts') {
            steps {
                archiveArtifacts artifacts: '**/*.pkl', fingerprint: true

                archiveArtifacts artifacts: '**/*.json', fingerprint: true

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
