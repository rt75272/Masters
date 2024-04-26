function get_user_input()
    current_time = input()
    alarm_duration = input()
    return current_time, alarm_duration

function set_alarm(time, alarm_length)
    alarm_target = (time + alarm_length) % 24

function main()
    times = get_user_input()
    set_alarm(times[0], times[1])

    