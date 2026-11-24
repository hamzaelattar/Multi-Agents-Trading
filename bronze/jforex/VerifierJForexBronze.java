import com.dukascopy.api.*;
import com.dukascopy.api.system.ClientFactory;
import com.dukascopy.api.system.IClient;
import java.io.BufferedWriter;
import java.io.Console;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/** Fixed read-only Bronze checks on a JForex DEMO session. */
public class VerifierJForexBronze implements IStrategy {
    private static final String DEMO = "https://platform.dukascopy.com/demo_3/jforex_3.jnlp";
    private static final DateTimeFormatter FORMAT = new DateTimeFormatterBuilder().appendInstant(3).toFormatter();
    private static final long HOUR = TimeUnit.HOURS.toMillis(1);
    private static final String[] DAYS = {"2021-01-05", "2022-07-05", "2007-03-20", "2007-04-10", "2025-08-05"};
    private static final int[] MISSING = {1, 3, 4, 5, 7, 8};
    private final Path output;
    private final String[] days;
    private final CountDownLatch done = new CountDownLatch(1);
    private volatile Exception failure;

    VerifierJForexBronze(Path output, String[] days) { this.output = output; this.days = days; }

    @Override public void onStart(IContext context) throws JFException {
        try {
            context.setSubscribedInstruments(Collections.singleton(Instrument.XAUUSD), true);
            Files.createDirectories(output);
            try (BufferedWriter journal = Files.newBufferedWriter(output.resolve("heures.tsv"), StandardCharsets.UTF_8,
                    java.nio.file.StandardOpenOption.CREATE_NEW)) {
                journal.write("date_utc\theure_utc\tstatut\tticks\n");
                journal.flush();
                for (String date : days) exportDay(context.getHistory(), journal, date);
            }
        } catch (Exception e) {
            failure = e;
            throw new JFException("Bronze verification stopped", e);
        } finally {
            done.countDown();
            context.stop();
        }
    }

    private void exportDay(IHistory history, BufferedWriter journal, String date) throws Exception {
        long origin = Instant.parse(date + "T00:00:00Z").toEpochMilli();
        int[] hours = date.equals("2021-01-05") ? MISSING : java.util.stream.IntStream.range(0, 24).toArray();
        Path pending = output.resolve(date + ".part.csv");
        Path finished = output.resolve(date + ".csv");
        if (Files.exists(finished)) throw new IOException("Output already exists for " + date);
        try (BufferedWriter writer = Files.newBufferedWriter(pending, StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.CREATE_NEW)) {
            writer.write("timestamp_utc,ask,bid,ask_volume,bid_volume\n");
            for (int hour : hours) {
                long start = origin + HOUR * hour;
                List<ITick> ticks;
                try {
                    ticks = history.getTicks(Instrument.XAUUSD, start, start + HOUR - 1);
                } catch (JFException e) {
                    record(journal, date, hour, "sdk_error", 0);
                    if (String.valueOf(e.getMessage()).matches("(?s).*\\b(403|429)\\b.*")) throw e;
                    Thread.sleep(2200);
                    continue;
                }
                for (ITick tick : ticks) {
                    if (tick.getTime() < start || tick.getTime() >= start + HOUR) {
                        throw new IOException("Tick outside requested hour");
                    }
                    writer.write(FORMAT.format(Instant.ofEpochMilli(tick.getTime())));
                    for (double value : new double[] {tick.getAsk(), tick.getBid(), tick.getAskVolume(), tick.getBidVolume()}) {
                        writer.write(',');
                        writer.write(BigDecimal.valueOf(value).toPlainString());
                    }
                    writer.write('\n');
                }
                writer.flush();
                record(journal, date, hour, ticks.isEmpty() ? "empty_unverified" : "downloaded", ticks.size());
                Thread.sleep(1500);
            }
        }
        Files.move(pending, finished);
    }

    private static void record(BufferedWriter journal, String date, int hour, String state, int ticks) throws IOException {
        String entry = date + "\t" + hour + "\t" + state + "\t" + ticks;
        journal.write(entry + "\n");
        journal.flush();
        System.out.println("BRONZE_HOUR " + entry);
    }

    public static void main(String[] args) throws Exception {
        Console console = System.console();
        if (args.length < 1 || args.length > 2 || console == null) {
            throw new IllegalArgumentException("Use a terminal with an output directory and optionally one finished UTC day.");
        }
        String[] selected = DAYS;
        if (args.length == 2) {
            java.time.LocalDate day = java.time.LocalDate.parse(args[1]);
            if (!day.isBefore(java.time.LocalDate.now(java.time.ZoneOffset.UTC))) {
                throw new IllegalArgumentException("The explicit UTC day must be finished.");
            }
            selected = new String[] {day.toString()};
        }
        char[] login = console.readPassword("JForex DEMO login: ");
        char[] password = console.readPassword("JForex DEMO password: ");
        if (login == null || password == null || login.length == 0 || password.length == 0) throw new IllegalArgumentException("DEMO credentials are required.");
        IClient client = ClientFactory.getDefaultInstance();
        try {
            try {
                client.connect(DEMO, new String(login), new String(password));
            } finally {
                Arrays.fill(login, '\0');
                Arrays.fill(password, '\0');
            }
            for (int i = 0; i < 120 && !client.isConnected(); i++) Thread.sleep(250);
            if (!client.isConnected()) throw new IOException("DEMO session unavailable.");
            VerifierJForexBronze check = new VerifierJForexBronze(Paths.get(args[0]).toAbsolutePath(), selected);
            long id = client.startStrategy(check);
            if (!check.done.await(32, TimeUnit.MINUTES)) {
                client.stopStrategy(id);
                throw new IOException("Bronze check timed out.");
            }
            if (check.failure != null) throw check.failure;
        } finally {
            client.disconnect();
        }
    }

    @Override public void onTick(Instrument instrument, ITick tick) {}
    @Override public void onBar(Instrument instrument, Period period, IBar ask, IBar bid) {}
    @Override public void onMessage(IMessage message) {}
    @Override public void onAccount(IAccount account) {}
    @Override public void onStop() {}
}
