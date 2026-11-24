import com.dukascopy.api.IAccount;
import com.dukascopy.api.IBar;
import com.dukascopy.api.IContext;
import com.dukascopy.api.IHistory;
import com.dukascopy.api.IMessage;
import com.dukascopy.api.IStrategy;
import com.dukascopy.api.ITick;
import com.dukascopy.api.Instrument;
import com.dukascopy.api.JFException;
import com.dukascopy.api.Period;
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
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/** Exports a single, fixed UTC day using IHistory. No trading API is used. */
public class PiloteXauusdJForex implements IStrategy {
    private static final long START = Instant.parse("2021-01-05T00:00:00Z").toEpochMilli();
    private static final long END = Instant.parse("2021-01-06T00:00:00Z").toEpochMilli();
    private static final long HOUR = TimeUnit.HOURS.toMillis(1);
    private static final DateTimeFormatter FORMAT = new DateTimeFormatterBuilder().appendInstant(3).toFormatter();
    private static final String DEMO = "https://platform.dukascopy.com/demo_3/jforex_3.jnlp";

    private final Path outputDirectory;
    private final CountDownLatch finished = new CountDownLatch(1);
    private volatile Exception failure;

    public PiloteXauusdJForex() {
        this(null);
    }

    private PiloteXauusdJForex(Path outputDirectory) {
        this.outputDirectory = outputDirectory;
    }

    @Override
    public void onStart(IContext context) throws JFException {
        try {
            Path directory = outputDirectory != null ? outputDirectory : context.getFilesDir().toPath();
            Path output = export(context.getHistory(), directory);
            context.getConsole().getOut().println("JForex pilot saved: " + output);
        } catch (IOException | RuntimeException | JFException e) {
            failure = e;
            throw new JFException("JForex pilot aborted", e);
        } finally {
            finished.countDown();
            context.stop();
        }
    }

    private static Path export(IHistory history, Path directory) throws IOException, JFException {
        Files.createDirectories(directory);
        Path finalCsv = directory.resolve("XAUUSD_2021-01-05_JForex_UTC.csv");
        Path pending = directory.resolve("XAUUSD_2021-01-05_JForex_UTC.part.csv");
        if (Files.exists(finalCsv)) throw new IOException("Output already exists: " + finalCsv);

        long count = 0;
        try (BufferedWriter writer = Files.newBufferedWriter(pending, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
            writer.write("timestamp_utc,ask,bid,ask_volume,bid_volume");
            writer.newLine();
            for (long start = START; start < END; start += HOUR) {
                long end = start + HOUR;
                List<ITick> ticks = history.getTicks(Instrument.XAUUSD, start, end - 1);
                for (ITick tick : ticks) {
                    if (tick.getTime() < start || tick.getTime() >= end) {
                        throw new IOException("Tick timestamp outside the requested hour");
                    }
                    writer.write(FORMAT.format(Instant.ofEpochMilli(tick.getTime())));
                    for (double value : new double[] {tick.getAsk(), tick.getBid(),
                            tick.getAskVolume(), tick.getBidVolume()}) {
                        writer.write(',');
                        writer.write(BigDecimal.valueOf(value).toPlainString());
                    }
                    writer.newLine();
                    count++;
                }
            }
        }
        if (count == 0) throw new IOException("Empty result. Pilot coverage remains unverified.");
        return Files.move(pending, finalCsv);
    }

    public static void main(String[] args) throws Exception {
        Console console = System.console();
        if (args.length != 1 || console == null) {
            throw new IllegalArgumentException("Run from a terminal with one output directory argument.");
        }
        String user = console.readLine("JForex DEMO login: ");
        char[] password = console.readPassword("JForex DEMO password: ");
        if (user == null || user.trim().isEmpty() || password == null || password.length == 0) {
            throw new IllegalArgumentException("A DEMO login is required.");
        }
        IClient client = ClientFactory.getDefaultInstance();
        try {
            try {
                client.connect(DEMO, user, new String(password));
            } finally {
                Arrays.fill(password, '\0');
            }
            for (int i = 0; i < 120 && !client.isConnected(); i++) Thread.sleep(250);
            if (!client.isConnected()) throw new IOException("DEMO session unavailable.");
            PiloteXauusdJForex pilot = new PiloteXauusdJForex(Paths.get(args[0]).toAbsolutePath());
            long id = client.startStrategy(pilot);
            if (!pilot.finished.await(10, TimeUnit.MINUTES)) {
                client.stopStrategy(id);
                throw new IOException("Pilot timed out.");
            }
            if (pilot.failure != null) throw pilot.failure;
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
