FROM spacevim/spacevim

USER root

# Install curl and other dependencies
RUN apt-get update && apt-get install -y curl

# Install locales and generate en_US.UTF-8
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Install Node.js and npm
RUN curl -fsSL https://deb.nodesource.com/setup_14.x | bash - && \
    apt-get install -y nodejs

# Install neovim npm package
RUN npm install -g neovim
RUN npm install -g devicons
RUN npm -g install remark
RUN npm -g install remark-cli
RUN npm -g install remark-stringify
RUN npm -g install remark-frontmatter
RUN npm -g install wcwidth
RUN npm install --global prettier
RUN apt-get update && apt-get install -y xclip

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install PHP without prompts
RUN apt-get update && \
    apt-get install -y --no-install-recommends php php-mysql php-xml php-mbstring && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3-pip && pip3 install notedown

# Switch back to spacevim user
USER spacevim

