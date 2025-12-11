{
  aspects = {
    jetstream = {
      message_handling = {
        ack = true,
        data_access = "Data()",
        log = true,
      },
    },
    prometheus = {
      metrics = {
        latency = true,
        errors = true,
        messages_processed = true,
      },
    },
  },
  hooks = {
    pre_process = "log.Printf('Pre-processing message: %v', msg.Subject())",
    post_process = "log.Printf('Post-processing message: %v', msg.Subject())",
  },
}
