pipeline {
  agent {
    node { label "docker-host" }
  }

  environment {
    GIT_NAME  = "rag-facts-check"
    registry  = "eeacms/rag-facts-check"
    // Rancher catalog path - set once a catalog entry exists for this image.
    template  = ""
    DEPENDENT_DOCKERFILE_URL = ""
  }

  stages {

    stage('Lint & Test') {
      when {
        allOf {
          not { buildingTag() }
          environment name: 'CHANGE_ID', value: ''
        }
      }
      steps {
        script {
          // The EEA Jenkins docker daemon is remote, so `docker run -v $PWD`
          // bind mounts are not visible to it. Bake code + tests into an
          // image (Dockerfile `test` stage) and copy the junit report out.
          def img = "$registry:ci-${env.BUILD_NUMBER}"
          def container = "${GIT_NAME}-ci-${env.BUILD_NUMBER}"
          try {
            sh "docker build --no-cache --target test -t ${img} ."

            // Lint / format: informational only - printed in the log, never
            // fails or marks the build unstable. Enforcement is left to the
            // pre-commit hook (scripts/hooks/pre-commit).
            echo '--- ruff (informational, non-blocking) ---'
            sh "docker run --rm --name='${container}-lint' ${img} sh -c 'ruff check rag_facts_check/ tests/ scripts/ || true; ruff format --check rag_facts_check/ tests/ scripts/ || true'"

            // Tests: hard failure, but always pull the junit report out first.
            def rc = sh(returnStatus: true, script: "docker run --name='${container}' ${img} pytest --junitxml=/app/junit.xml")
            sh "docker cp '${container}:/app/junit.xml' junit.xml || true"
            if (rc != 0) {
              error("pytest failed (exit ${rc})")
            }
          } finally {
            sh "docker rm -f '${container}' || true"
            sh "docker rmi ${img} || true"
          }
        }
      }
      post {
        always {
          junit testResults: 'junit.xml', allowEmptyResults: true
        }
      }
    }

    stage('Docker build & push ( on tag )') {
      when {
        buildingTag()
      }
      steps {
        script {
          // Build the runtime image and push it as :<git-tag> and :latest.
          // Mirrors eea/cca-frontend. The eeacms/gitflow Release stage below
          // also publishes, but pushing here keeps the tagged image available
          // even if the catalog/release step is skipped or fails.
          // On a tag build BRANCH_NAME is the tag name.
          def imageTag = env.TAG_NAME ?: env.BRANCH_NAME
          try {
            def image = docker.build("${registry}:${imageTag}", "--no-cache .")
            docker.withRegistry('', 'eeajenkins') {
              image.push()
              image.push('latest')
            }
          } finally {
            sh "docker rmi ${registry}:${imageTag} || true"
          }
        }
      }
    }

    stage('Release ( on tag )') {
      when {
        buildingTag()
      }
      steps {
        node(label: 'docker') {
          // eeacms/gitflow builds + pushes the Docker image, creates the GitHub
          // release, and (when `template` is set) bumps the Rancher catalog.
          // Needs the eeacms/rag-facts-check Docker Hub repo to exist with push
          // rights for the eeajenkins credential.
          withCredentials([string(credentialsId: 'eea-jenkins-token', variable: 'GITHUB_TOKEN'), usernamePassword(credentialsId: 'jekinsdockerhub', usernameVariable: 'DOCKERHUB_USER', passwordVariable: 'DOCKERHUB_PASS')]) {
            sh '''docker pull eeacms/gitflow; docker run -i --rm --name="$BUILD_TAG-release" \
              -e GIT_BRANCH="$BRANCH_NAME" \
              -e GIT_NAME="$GIT_NAME" \
              -e DOCKERHUB_REPO="$registry" \
              -e GIT_TOKEN="$GITHUB_TOKEN" \
              -e DOCKERHUB_USER="$DOCKERHUB_USER" \
              -e DOCKERHUB_PASS="$DOCKERHUB_PASS" \
              -e DEPENDENT_DOCKERFILE_URL="$DEPENDENT_DOCKERFILE_URL" \
              -e RANCHER_CATALOG_PATHS="$template" \
              -e GITFLOW_BEHAVIOR="RUN_ON_TAG" \
              eeacms/gitflow'''
          }
        }
      }
    }

  }

  post {
    always {
      cleanWs(cleanWhenAborted: true, cleanWhenFailure: true, cleanWhenNotBuilt: true, cleanWhenSuccess: true, cleanWhenUnstable: true, deleteDirs: true)
    }
    changed {
      script {
        def url = "${env.BUILD_URL}/display/redirect"
        def status = currentBuild.currentResult
        def details = """<h1>${env.JOB_NAME} - Build #${env.BUILD_NUMBER} - ${status}</h1>
                         <p>Check console output at <a href="${url}">${env.JOB_BASE_NAME} - #${env.BUILD_NUMBER}</a></p>
                      """
        emailext(
          subject: '$DEFAULT_SUBJECT',
          body: details,
          attachLog: true,
          compressLog: true,
          recipientProviders: [[$class: 'DevelopersRecipientProvider'], [$class: 'CulpritsRecipientProvider']]
        )
      }
    }
  }
}
