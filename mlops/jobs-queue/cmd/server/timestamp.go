package main

import (
	"fmt"
	"time"
)

func TimeStamp() string {
	currentTime := time.Now()
	return fmt.Sprintf(
		"%d-%d-%d %d:%d:%d",
		currentTime.Year(),
		currentTime.Month(),
		currentTime.Day(),
		currentTime.Hour(),
		currentTime.Hour(),
		currentTime.Second(),
	)
}
