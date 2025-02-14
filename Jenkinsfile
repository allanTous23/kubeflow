pipeline {
    agent any

    environment {
        CLEAN_IMAGE = 'clean-data:latest'
    }

    stages {
        stage('creation repertoire travail') {
            steps {
             bat 'if not exist model_artifacts mkdir model_artifacts'

            }
        }
        
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
                archiveArtifacts artifacts: '/*.pkl', fingerprint: true
            }
        }

        // stage('Démarrer les services') {
        //     steps {
        //         bat 'docker-compose up -d'
        //     }
        // }
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
