all:
	@nohup konsole -e htop &
	@sleep 5s
	@pkill htop
	@echo "\nscreenfetch" | figlet | lolcat
	@screenfetch | lolcat
	@echo "\nlsblk" | figlet | lolcat
	@lsblk | lolcat
	@echo "\npydf" | figlet | lolcat
	@pydf | lolcat
	@rm nohup.out

